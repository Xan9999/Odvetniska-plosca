import json
import os
import uuid
import pathlib
from datetime import datetime

import io

import pypdf
import requests
from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from openai import OpenAI
from pydantic import BaseModel
from rapidfuzz import fuzz

load_dotenv()

app = FastAPI(title="Jadek & Pensa — Sistem za sprejem strank")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
app.mount("/static", StaticFiles(directory="static"), name="static")

openai_client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

TFL_API_KEY = os.environ.get("TFL_API_KEY", "")
TFL_BASE = "https://www.tax-fin-lex.si/api/v1"
TFL_HEADERS = {"X-Api-Key": TFL_API_KEY, "Content-Type": "application/json"}

DB = pathlib.Path("db")
EMAILS_FILE = DB / "emails.json"

LEGAL_FIELDS = [
    "DELOVNO PRAVO",
    "BANČNIŠTVO IN FINANCE",
    "DAVČNO PRAVO",
    "ENERGETIKA",
    "TEHNOLOGIJA, MEDIJI IN ELEKTRONSKE KOMUNIKACIJE",
    "INSOLVENČNO PRAVO IN PRESTRUKTURIRANJA",
    "INTELEKTUALNA LASTNINA",
    "JAVNO NAROČANJE",
    "KOMERCIALNE POGODBE",
    "KONKURENČNO PRAVO",
    "KORPORACIJSKO PRAVO",
    "MIGRACIJSKO PRAVO",
    "NALOŽBENI SKLADI",
    "NEPREMIČNINE, GRADBENIŠTVO IN INFRASTRUKTURA",
    "PREPREČEVANJE IN REŠEVANJE SPOROV",
    "PREVZEMI IN ZDRUŽITVE",
    "REGULACIJA S PODROČJA ZDRAVIL",
    "VARSTVO OSEBNIH PODATKOV",
]


# ── DB helpers ────────────────────────────────────────────────────────────────

def load_db(filename: str):
    return json.loads((DB / filename).read_text(encoding="utf-8"))


def get_emails() -> list:
    if not EMAILS_FILE.exists():
        return []
    return json.loads(EMAILS_FILE.read_text(encoding="utf-8"))


def save_emails(emails: list):
    EMAILS_FILE.write_text(json.dumps(emails, ensure_ascii=False, indent=2), encoding="utf-8")


# ── LLM calls ─────────────────────────────────────────────────────────────────

def llm_analyze(email_text: str) -> dict:
    fields_list = "\n".join(f"- {f}" for f in LEGAL_FIELDS)
    system = f"""Si pravni asistent v odvetniški pisarni Jadek & Pensa.
Analiziraj dohodni e-mail potencialne stranke in vrni SAMO veljavni JSON, brez dodatnega besedila.

Razpoložljiva pravna področja (izberi ENO najprimernejše):
{fields_list}

Vrni ta JSON:
{{
  "jezik": "<sl|en|sl-en — DOLOČI NAJPREJ, preden napišeš karkoli drugega>",
  "legal_field": "<eno od zgoraj naštetih področij, dobesedno>",
  "nujnost": <0.0–1.0, kjer 1.0 = izjemno nujno>,
  "profit": <ocena zaslužka v EUR, celo število>,
  "rok": {{
    "nujno": <true/false>,
    "cas_dni": <število dni ali null>,
    "opis": "<kratek opis roka ali null>"
  }},
  "entitete": {{
    "stranke": ["<ime podjetja ali osebe — stranka ki piše>"],
    "nasprotna_stran": ["<ime podjetja ali osebe na nasprotni strani>"],
    "ostali_upleteni": ["<ostale relevantne entitete>"]
  }},
  "zahtevnost": <1–10, kjer 10 = izjemno kompleksno>,
  "zahtevnost_razlaga": "<1–2 stavka: zakaj takšna ocena zahtevnosti>",
  "povzetek": "<3–5 stavkov strnjen povzetek z pravno terminologijo>",
  "dodatna_vprasanja": ["<vprašanje 1 — najpomembnejše, V JEZIKU MAILA>", "<vprašanje 2>", "<vprašanje 3>", "<vprašanje 4>", "<vprašanje 5>"],
  "aml_tveganje": "<nizko|srednje|visoko>",
  "aml_razlaga": "<1–2 stavka: zakaj takšna ocena AML tveganja — navedi konkretne indikatorje: jurisdikcija, izvor sredstev, lastniška struktura, vrsta posla>",
  "aml_checklist": ["<dokument 1 — V JEZIKU MAILA>", "<dokument 2>", ...]
}}

⚠️ JEZIK — OBVEZNO: Polji "dodatna_vprasanja" IN "aml_checklist" morata biti IZKLJUČNO v ISTEM JEZIKU KOT E-MAIL.
- Mail v slovenščini → oba polji SAMO v slovenščini, nobena beseda v angleščini
- Mail v angleščini → oba polji SAMO v angleščini
- Mešan mail (sl-en) → slovenščina

Pri aml_checklist upoštevaj konkretne okoliščine zadeve (ne le splošni seznam). Osnova glede na tveganje:
- nizko: identity document / court registry extract, UBO declaration
- srednje: + proof of source of funds, description of business relationship, ownership structure chart
- visoko: + bank statements / proof of wealth, senior management approval, ongoing monitoring consent, foreign registry certificate
Dodaj specifične postavke glede na dejansko zadevo (npr. pri M&A: financial statements of target company, pri nepremičninah: land registry extract).
Vsak element je konkreten dokument ali ukrep z imenom entitete (npr. "Court registry extract — Rheingold Capital GmbH")."""

    resp = openai_client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": f"E-mail:\n\n{email_text}\n\nIMPORTANT: Detect the email language first (set \"jezik\" field), then write \"dodatna_vprasanja\" and \"aml_checklist\" in THAT SAME language."},
        ],
        response_format={"type": "json_object"},
        temperature=0.2,
    )
    return json.loads(resp.choices[0].message.content)


def llm_select_cases(analysis: dict, all_cases: list) -> list[str]:
    relevant = [c for c in all_cases if analysis.get("legal_field") in c.get("fields", [])]
    if not relevant:
        relevant = all_cases

    cases_summary = "\n".join(
        f"ID: {c['id']} | {c['title']} ({c['year']}) | {c['short_description']}"
        for c in relevant
    )

    system = """Si pravni asistent. Izberi 2–3 pretekle primere iz baze, ki so najbolj relevantni za zadevo in jih odvetnik lahko prepričljivo citira v odgovoru stranki.
Vrni SAMO JSON: {"selected_ids": ["id1", "id2"]}"""

    user = f"""Zadeva: {analysis.get('povzetek', '')}
Pravno področje: {analysis.get('legal_field', '')}

Razpoložljivi primeri:
{cases_summary}"""

    resp = openai_client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        response_format={"type": "json_object"},
        temperature=0.1,
    )
    result = json.loads(resp.choices[0].message.content)
    return result.get("selected_ids", [])


def llm_generate_draft(email_text: str, analysis: dict, selected_cases: list) -> dict:
    cases_text = "\n\n".join(
        f"[{c['id']}] {c['title']} ({c['year']})\n{c['description']}\nIzid: {c['outcome']}"
        for c in selected_cases
    ) if selected_cases else "Ni razpoložljivih primerov."

    system = """Si izkušen odvetnik v pisarni Jadek & Pensa. Napiši profesionalen osnutek odgovora na e-mail potencialne stranke.

━━ OBVEZNI PLACEHOLDERJI — vstavi jih TOČNO takšne, brez kakršnih koli sprememb ━━

1. EKIPA: Kjer v besedilu navedete predlagano ekipo, napišite TOČNO: {{PREDLAGANA_EKIPA}}
   Primer: "Vaš primer bomo zaupali ekipi {{PREDLAGANA_EKIPA}}, ki ima bogate izkušnje..."
   NIKOLI ne pišite imen odvetnikov direktno. Vedno samo {{PREDLAGANA_EKIPA}}.

2. ODGOVORI NA VPRAŠANJA: Če stranka v mailu postavlja konkretna vprašanja (oštevilčena ali eksplicitna),
   jim odgovori s placeholderjem {{QA_ODGOVORI}} na mestu kjer bodo odgovori.
   Primer: "Z veseljem odgovarjamo na vaša vprašanja:\n{{QA_ODGOVORI}}"
   Posameznih vprašanj ne odgovarjaj v main besedilu — samo {{QA_ODGOVORI}}.
   Če v mailu NI konkretnih vprašanj, tega placeholderja ne vključi.

3. NAŠA VPRAŠANJA: Na koncu (pred podpisom) dodaj uvodni stavek V JEZIKU MAILA, nato placeholder:
   {{USMERJEVALNA_VPRASANJA}}
   Primer (sl): "Za učinkovit začetek vas prosimo za odgovor na naslednja vprašanja:\n{{USMERJEVALNA_VPRASANJA}}"
   Primer (en): "To help us get started, please answer the following questions:\n{{USMERJEVALNA_VPRASANJA}}"

4. POTREBNA DOKUMENTACIJA: Takoj za vprašanji (pred podpisom) dodaj uvodni stavek V JEZIKU MAILA, nato placeholder:
   {{POTREBNA_DOKUMENTACIJA}}
   Primer (sl): "Za obravnavo zadeve bomo potrebovali naslednjo dokumentacijo:\n{{POTREBNA_DOKUMENTACIJA}}"
   Primer (en): "To proceed with your matter, we will require the following documentation:\n{{POTREBNA_DOKUMENTACIJA}}"

5. PODPIS: [Vaše ime]

━━ NAVAJANJE PRIMEROV ━━
Vstavi [REF:case_id] TAKOJ za poved ki citira izkušnje pisarne (pred presledkom, za morebitno piko).
Primer: "Uspešno smo zavarovali patent za algoritem slovenskega tech podjetja.[REF:ip-2023-01]"
Citiraj SAMO podane primere. Vsak največ enkrat.

━━ FORMAT ODGOVORA — vrni SAMO ta JSON ━━
{
  "draft": "<celoten osnutek z vsemi placeholderji>",
  "qa_blocks": [
    {
      "vprasanje": "<dobesedno vprašanje stranke>",
      "odgovor": "<natančen pravni odgovor, 2–4 stavki>",
      "vir_namig": null
    }
  ],
  "citations": {
    "REF:case_id": {
      "naslov": "<naslov primera>",
      "leto": <leto>,
      "opis": "<1–2 stavka>",
      "izid": "<kratek izid>"
    }
  }
}

Če ni konkretnih vprašanj stranke, "qa_blocks" je prazna lista [].
Ton: profesionalen, samozavesten, topel. Jezik: slovenščina (razen če je mail v angleščini).
Vrstni red v draftu: uvod → razumevanje → izkušnje s citati → {{QA_ODGOVORI}} (če primerno) → ekipa → naslednji koraki → {{USMERJEVALNA_VPRASANJA}} → {{POTREBNA_DOKUMENTACIJA}} → [Vaše ime]"""

    jezik = analysis.get('jezik', 'sl')
    jezik_navodilo = {
        'en':    'LANGUAGE: The email is in English. Write the entire draft, all questions ({{USMERJEVALNA_VPRASANJA}}), and documentation list ({{POTREBNA_DOKUMENTACIJA}}) in ENGLISH.',
        'sl-en': 'LANGUAGE: The email is bilingual (SLO/ENG). Write the entire draft in Slovenian.',
    }.get(jezik, 'LANGUAGE: The email is in Slovenian. Write the entire draft in Slovenian.')

    user = f"""ORIGINALNI E-MAIL:
{email_text}

ANALIZA:
Področje: {analysis.get('legal_field')}
Povzetek: {analysis.get('povzetek')}
Zahtevnost: {analysis.get('zahtevnost')}/10

{jezik_navodilo}

RELEVANTNI PRIMERI IZ BAZE:
{cases_text}"""

    resp = openai_client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        response_format={"type": "json_object"},
        temperature=0.4,
    )
    return json.loads(resp.choices[0].message.content)


# ── Tax-Fin-Lex integration ───────────────────────────────────────────────────

def tfl_ask(question: str, max_tokens: int = 800) -> dict:
    """Call TFL /ai/ask, collect SSE stream, return {answer, sources}."""
    if not TFL_API_KEY:
        return {"answer": "", "sources": []}
    try:
        resp = requests.post(
            f"{TFL_BASE}/ai/ask",
            headers={**TFL_HEADERS, "Accept": "text/event-stream"},
            json={"question": question, "maxTokens": max_tokens},
            stream=True,
            timeout=60,
        )
        tokens, sources = [], []
        for raw in resp.iter_lines():
            line = raw.decode("utf-8") if isinstance(raw, bytes) else raw
            if not line.startswith("data: "):
                continue
            data_str = line[6:]
            if data_str == "[DONE]":
                break
            try:
                ev = json.loads(data_str)
            except json.JSONDecodeError:
                continue
            t = ev.get("type")
            if t == "token":
                tokens.append(ev["data"]["text"])
            elif t == "sources":
                sources = ev["data"]

        full = "".join(tokens)
        # Strip "## Razmišljanje" preamble — keep only the final answer
        if "## Končni odgovor:" in full:
            full = full.split("## Končni odgovor:", 1)[1].strip()
        # Strip trailing "Ključni dokumenti" section (we show sources separately)
        for marker in ["#### Ključni dokumenti", "### Ključni dokumenti", "**Ključni dokumenti"]:
            if marker in full:
                full = full.split(marker, 1)[0].strip()
        return {"answer": full, "sources": sources[:5]}  # max 5 sources
    except Exception:
        return {"answer": "", "sources": []}


def tfl_answer_questions(questions: list, legal_field: str) -> list:
    """For each question ask TFL and return qa_blocks with real sources."""
    qa_blocks = []
    for q in questions[:3]:  # max 3 to stay within rate limits
        prompt = (
            f"{q} (področje: {legal_field}). "
            "Odgovori jedrnato v 3–5 stavkih brez emotikonov in brez razdelkov s poudarjenimi naslovi."
        )
        result = tfl_ask(prompt, max_tokens=400)
        qa_blocks.append({
            "vprasanje": q,
            "odgovor": result["answer"],
            "sources": result["sources"],
        })
    return qa_blocks


_TFL_URL_MAP = {
    "legislation": "/Zakonodaja/Podrobnosti/{}",
    "court":       "/SodnaPraksa/Podrobnosti/{}",
    "publication": "/Publikacije/Podrobnosti/{}",
}


def _tfl_fix_url(item: dict) -> dict:
    """Rewrite API-path URLs to correct TFL web URLs using entityId + type."""
    entity_id = item.get("id") or item.get("entityId", "")
    item_type = item.get("type", "")
    if entity_id and item_type in _TFL_URL_MAP:
        item["url"] = _TFL_URL_MAP[item_type].format(entity_id)
    return item


def tfl_search(query: str, types: list | None = None) -> list:
    """Semantic search over TFL legal database."""
    if not TFL_API_KEY:
        return []
    try:
        payload = {"query": query}
        if types:
            payload["types"] = types
        resp = requests.post(
            f"{TFL_BASE}/search",
            headers=TFL_HEADERS,
            json=payload,
            timeout=20,
        )
        data = resp.json()
        items = data.get("data", {}).get("items", [])[:5]
        return [_tfl_fix_url(i) for i in items]
    except Exception:
        return []


# ── Attachment extraction ─────────────────────────────────────────────────────

def extract_pdf_text(data: bytes) -> str:
    try:
        reader = pypdf.PdfReader(io.BytesIO(data))
        pages = [page.extract_text() or "" for page in reader.pages]
        return "\n\n".join(p for p in pages if p.strip())
    except Exception:
        return ""


# ── Conflict check ─────────────────────────────────────────────────────────────

CONFLICT_THRESHOLD = 80


def check_conflict(entities: dict, clients: list) -> dict:
    conflicts = []      # nasprotna_stran matches existing client → RED
    risks = []          # ostali_upleteni matches existing client → ORANGE
    returning = []      # stranke matches existing client → OK (returning client)

    for role, names in entities.items():
        for name in names:
            for client in clients:
                score = fuzz.token_sort_ratio(name.lower(), client["name"].lower())
                if score >= CONFLICT_THRESHOLD:
                    entry = {
                        "entity": name,
                        "role": role,
                        "matched_client": client["name"],
                        "client_id": client["id"],
                        "score": score,
                        "active": client.get("active", True),
                    }
                    if role == "nasprotna_stran":
                        conflicts.append(entry)
                    elif role == "ostali_upleteni":
                        risks.append(entry)
                    else:
                        # stranke = returning client, not a conflict
                        returning.append(entry)

    if conflicts:
        status, color = "konflikt", "red"
    elif risks:
        status, color = "tveganje", "orange"
    else:
        status, color = "ok", "green"

    return {
        "status": status,
        "color": color,
        "conflicts": conflicts,
        "risks": risks,
        "returning": returning,
    }


# ── Team / attorney selection ──────────────────────────────────────────────────

WORKLOAD_LABELS = {
    1.0: ("Prosta kapaciteta", "green"),
    1.5: ("Prosta kapaciteta", "green"),
    2.0: ("Normalna obremenitev", "orange"),
    2.5: ("Normalna obremenitev", "orange"),
    3.0: ("Nadure", "red"),
}


def get_field_attorneys(legal_field: str, lf_db: dict, att_db: list) -> list:
    """Return ALL attorneys for the given field, sorted by workload (ascending)."""
    acronyms = lf_db.get(legal_field, [])
    att_map = {a["acronym"]: a for a in att_db}

    team = []
    for acr in acronyms:
        if acr in att_map:
            a = dict(att_map[acr])
            label, color = WORKLOAD_LABELS.get(a["workload"], ("Neznano", "gray"))
            a["workload_label"] = label
            a["workload_color"] = color
            team.append(a)

    # TODO: also sort by seniority, hourly rate, past experience in this field
    team.sort(key=lambda x: x["workload"])
    return team


# ── Routes ────────────────────────────────────────────────────────────────────

class EmailIn(BaseModel):
    sender_name: str
    sender_email: str
    subject: str
    body: str

MAX_ATTACHMENT_CHARS = 12_000  # ~3k tokens per attachment


@app.get("/")
def serve_index():
    return FileResponse("static/index.html")


@app.get("/detail")
def serve_detail():
    return FileResponse("static/detail.html")


@app.post("/api/emails")
async def process_email(
    sender_name:  str = Form(...),
    sender_email: str = Form(...),
    subject:      str = Form(...),
    body:         str = Form(...),
    attachments:  list[UploadFile] = File(default=[]),
):
    attorneys = load_db("attorneys.json")
    legal_fields = load_db("legal_fields.json")
    cases = load_db("cases.json")
    clients = load_db("clients.json")

    # Extract text from PDF attachments
    attachment_texts = []
    attachment_meta = []
    for f in attachments:
        if not f.filename:
            continue
        raw = await f.read()
        fname = f.filename.lower()
        if fname.endswith(".pdf"):
            text = extract_pdf_text(raw)
        else:
            try:
                text = raw.decode("utf-8", errors="replace")
            except Exception:
                text = ""
        if text.strip():
            text = text[:MAX_ATTACHMENT_CHARS]
            attachment_texts.append(f"[PRILOGA: {f.filename}]\n{text}")
            attachment_meta.append({"filename": f.filename, "chars": len(text)})

    full_email = (
        f"Od: {sender_name} <{sender_email}>\n"
        f"Zadeva: {subject}\n\n"
        f"{body}"
    )
    if attachment_texts:
        full_email += "\n\n" + "\n\n".join(attachment_texts)

    # Call 1 — analyse
    analysis = llm_analyze(full_email)

    # Conflict check (Python, not LLM)
    conflict = check_conflict(analysis.get("entitete", {}), clients)

    # All attorneys for this field, sorted by workload
    all_field_attorneys = get_field_attorneys(analysis.get("legal_field", ""), legal_fields, attorneys)
    analysis["all_field_attorneys"] = all_field_attorneys

    # Suggested team = top N based on complexity (for draft generation reference)
    zahtevnost = analysis.get("zahtevnost", 5)
    cutoff = 3 if zahtevnost >= 6 else 2
    analysis["suggested_team"] = all_field_attorneys[:cutoff]

    # Call 2 — pick relevant cases
    selected_ids = llm_select_cases(analysis, cases)
    selected_cases = [c for c in cases if c["id"] in selected_ids]

    # Call 3 — generate draft (GPT-4o)
    draft_result = llm_generate_draft(full_email, analysis, selected_cases)

    # Call 4 — answer Q&A blocks with TFL (Gemini 2.5 Pro + Slovenian law DB)
    questions = analysis.get("dodatna_vprasanja", [])
    if TFL_API_KEY and questions:
        tfl_qa = tfl_answer_questions(questions, analysis.get("legal_field", ""))
    else:
        tfl_qa = draft_result.get("qa_blocks", [])

    # Call 5 — TFL semantic search for legal context
    tfl_refs = []
    if TFL_API_KEY and analysis.get("povzetek"):
        tfl_refs = tfl_search(analysis["povzetek"], types=["act", "court"])

    record = {
        "id": str(uuid.uuid4()),
        "created_at": datetime.now().isoformat(),
        "sender": {
            "name": sender_name,
            "email": sender_email,
            "subject": subject,
        },
        "body": body,
        "attachments": attachment_meta,
        "analysis": analysis,
        "conflict": conflict,
        "draft": draft_result.get("draft", ""),
        "citations": draft_result.get("citations", {}),
        "qa_blocks": tfl_qa,
        "tfl_refs": tfl_refs,
        "selected_cases": selected_cases,
        "status": "new",
    }

    emails = get_emails()
    emails.append(record)
    save_emails(emails)

    return record


@app.get("/api/emails")
def list_emails():
    emails = get_emails()
    emails.sort(key=lambda e: e.get("analysis", {}).get("profit", 0), reverse=True)
    return [
        {
            "id": e["id"],
            "created_at": e["created_at"],
            "sender": e["sender"],
            "legal_field": e.get("analysis", {}).get("legal_field", ""),
            "nujnost": e.get("analysis", {}).get("nujnost", 0),
            "profit": e.get("analysis", {}).get("profit", 0),
            "rok": e.get("analysis", {}).get("rok", {}),
            "zahtevnost": e.get("analysis", {}).get("zahtevnost", 5),
            "conflict_status": e.get("conflict", {}).get("status", "ok"),
            "aml_tveganje": e.get("analysis", {}).get("aml_tveganje", "nizko"),
            "povzetek": e.get("analysis", {}).get("povzetek", ""),
            "status": e.get("status", "new"),
        }
        for e in emails
    ]


@app.get("/api/emails/{email_id}")
def get_email(email_id: str):
    for e in get_emails():
        if e["id"] == email_id:
            return e
    raise HTTPException(status_code=404, detail="Email not found")


@app.patch("/api/emails/{email_id}/status")
def update_status(email_id: str, status: str):
    emails = get_emails()
    for e in emails:
        if e["id"] == email_id:
            e["status"] = status
            save_emails(emails)
            return {"ok": True}
    raise HTTPException(status_code=404, detail="Email not found")


@app.delete("/api/emails")
def delete_all_emails():
    save_emails([])
    return {"ok": True}


@app.get("/api/tfl/external-url")
def tfl_external_url(id: str, type: str = "legislation"):
    """Resolve a TFL entity ID to its canonical external URL (PISRS or sodnapraksa.si)."""
    if not TFL_API_KEY:
        raise HTTPException(status_code=503, detail="TFL API key not configured")
    try:
        if type == "court":
            resp = requests.get(f"{TFL_BASE}/court-decisions/{id}", headers=TFL_HEADERS, timeout=15)
            data = resp.json().get("data", {})
            ecli = data.get("ecli", "")
            if ecli:
                url = f"https://sodnapraksa.si/?q=ecliid:{ecli}"
            else:
                # Fallback: sodnapraksa search by document title
                title = data.get("title", "")
                url = f"https://sodnapraksa.si/?q={requests.utils.quote(title)}"
        else:
            resp = requests.get(f"{TFL_BASE}/legislation/{id}/metadata", headers=TFL_HEADERS, timeout=15)
            data = resp.json().get("data", {})
            sop = (data.get("sop") or "").strip()
            if sop:
                url = f"https://www.pisrs.si/Pis.web/pregledPredpisa?sop={sop}"
            else:
                abbr = (data.get("abbreviation") or "").strip()
                url = f"https://www.pisrs.si/Pis.web/pregled?besedilo={requests.utils.quote(abbr)}"
        return {"url": url}
    except Exception as e:
        raise HTTPException(status_code=502, detail=str(e))


@app.get("/api/tfl/doc")
def tfl_doc(id: str, type: str = "legislation"):
    """Fetch a TFL document (legislation or court decision) via API and return content."""
    if not TFL_API_KEY:
        raise HTTPException(status_code=503, detail="TFL API key not configured")
    try:
        if type == "court":
            resp = requests.get(f"{TFL_BASE}/court-decisions/{id}", headers=TFL_HEADERS, timeout=20)
            data = resp.json().get("data", {})
            return {
                "title": data.get("title", ""),
                "type": "court",
                "court": data.get("court", ""),
                "date": (data.get("documentDate") or "")[:10],
                "summary_q": data.get("summaryQuestion", ""),
                "summary_a": data.get("summaryAnswer", ""),
                "keywords": data.get("keywords", ""),
                "text": (data.get("text") or "")[:4000],  # cap at 4000 chars
            }
        else:
            resp = requests.get(f"{TFL_BASE}/legislation/{id}", headers=TFL_HEADERS, timeout=20)
            data = resp.json().get("data", {})
            articles = data.get("articles", [])
            return {
                "title": data.get("title", ""),
                "type": "legislation",
                "abbreviation": data.get("abbreviation", ""),
                "article_count": data.get("articleCount", 0),
                "articles": [
                    {"mark": a.get("mark", ""), "content": a.get("html", a.get("content", ""))}
                    for a in articles[:15]
                ],
            }
    except Exception as e:
        raise HTTPException(status_code=502, detail=str(e))


class TflAskIn(BaseModel):
    question: str
    max_tokens: int = 1500


@app.post("/api/tfl/ask")
def tfl_ask_proxy(body: TflAskIn):
    """Stream TFL /ai/ask response directly to frontend (SSE)."""
    if not TFL_API_KEY:
        raise HTTPException(status_code=503, detail="TFL API key not configured")

    def generate():
        try:
            resp = requests.post(
                f"{TFL_BASE}/ai/ask",
                headers={**TFL_HEADERS, "Accept": "text/event-stream"},
                json={"question": body.question, "maxTokens": body.max_tokens},
                stream=True,
                timeout=60,
            )
            for raw in resp.iter_lines():
                line = raw.decode("utf-8") if isinstance(raw, bytes) else raw
                if line:
                    yield line + "\n\n"
        except Exception as e:
            yield f'data: {{"type":"error","data":{{"message":"{str(e)}"}}}}\n\n'

    return StreamingResponse(generate(), media_type="text/event-stream")
