import json
import os
import re
import uuid
import pathlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, date, timedelta

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

AJPES_PRS_SEARCH = "https://prs.ajpes.si/prs/app/web/VpisList"
AJPES_PRS_DETAIL = "https://prs.ajpes.si/prs/app/web/Vpis"

app = FastAPI(title="Jadek & Pensa — Sistem za sprejem strank")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/favicon.ico", include_in_schema=False)
def favicon():
    # Serve favicon from repo root so browsers can GET /favicon.ico
    return FileResponse("favicon.ico", media_type="image/x-icon")

openai_client = OpenAI(
    api_key=os.environ["OPENAI_API_KEY"],
    base_url=os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"),
)

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
    "nujno": <true/false — ali obstaja kakršenkoli rok>,
    "stranka_cas_dni": <koliko dni ima stranka po lastnih besedah ali null>,
    "preteklo_dni": <koliko dni je že preteklo od sprožitvenega dogodka omenjenega v mailu ali null>,
    "opis": "<1 stavek: opis konteksta roka — kaj je sprožilni dogodek, kaj je stranka omenila>"
  }},
  "entitete": {{
    "stranke": ["<ime podjetja ali osebe — stranka ki piše>"],
    "nasprotna_stran": ["<ime podjetja ali osebe na nasprotni strani>"],
    "ostali_upleteni": ["<ostale relevantne entitete>"]
  }},
  "zahtevnost": <1–10, kjer 10 = izjemno kompleksno>,
  "zahtevnost_razlaga": "<1–2 stavka: zakaj takšna ocena zahtevnosti>",
  "povzetek": "<3–5 stavkov strnjen povzetek z pravno terminologijo>",
  "tip_stranke": "<fizicna|pravna — ali gre za fizično ali pravno osebo (d.o.o., d.d., GmbH …)>"
}}

⚠️ ROK: Ekstrahiraj SAMO kar piše v mailu — ne ugibaj zakonodaje. Zakonski rok bo določen posebej.
⚠️ Vrni SAMO JSON brez kakršnega koli besedila zunaj JSON objekta."""

    resp = openai_client.chat.completions.create(
        model="gpt-5.4",
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": f"E-mail:\n\n{email_text}"},
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
        model="gpt-5.4",
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

5. VIDEO KONFERENCA: V naslednje korake vključi predlog video konference z naslednjim placeholderjem:
   {{PREDLAGANI_TERMIN}}
   Primer (sl): "Predlagamo uvodni video posvet dne {{PREDLAGANI_TERMIN}}, na katerem bi podrobneje razpravljali o vaši zadevi."
   Primer (en): "We propose an initial video conference on {{PREDLAGANI_TERMIN}} to discuss your matter in detail."

6. PODPIS: [Vaše ime]

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
        model="gpt-5.4",
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        response_format={"type": "json_object"},
        temperature=0.4,
    )
    return json.loads(resp.choices[0].message.content)


# ── AJPES integration ────────────────────────────────────────────────────────

# High-risk jurisdictions for AML purposes (FATF grey/blacklist + known offshore)
HIGH_RISK_JURISDICTIONS = {
    "ae", "uae", "cayman", "bvi", "british virgin", "panama", "seychelles",
    "belize", "marshall islands", "liechtenstein", "andorra", "monaco",
    "iran", "north korea", "myanmar", "russia", "belarus", "syria",
    "afghanistan", "somalia", "yemen", "libya", "venezuela", "cuba",
    "nigeria", "kenya", "pakistan", "cambodia", "haiti", "south sudan",
}


def _ajpes_normalize(raw: dict) -> dict:
    """Normalize an AJPES record to a consistent internal format."""
    return {
        "ime": raw.get("ime") or raw.get("name") or raw.get("naziv") or "",
        "maticna": raw.get("maticnaStevilka") or raw.get("maticna") or "",
        "status": raw.get("statusSubjekta") or raw.get("status") or "",
        "naslov": raw.get("naslov") or raw.get("naslovUlica") or "",
        "dejavnost": raw.get("dejavnost") or raw.get("skdKoda") or "",
        "zastopniki": raw.get("zastopniki") or raw.get("directors") or [],
        "lastniki": raw.get("lastniki") or raw.get("owners") or [],
        "kapital": raw.get("osnovniKapital") or raw.get("kapital"),
        "ustanovljena": raw.get("datumVpisa") or raw.get("founded") or "",
        "pravnaoblika": raw.get("pravnaOblika") or raw.get("legalForm") or "",
    }


def ajpes_lookup(name: str) -> dict:
    """
    Look up an entity in AJPES PRS (Poslovni register Slovenije).
    Returns normalised dict or {} on failure/not found.
    """
    if not name or len(name.strip()) < 3:
        return {}
    try:
        resp = requests.get(
            AJPES_PRS_SEARCH,
            params={"format": "json", "ime": name.strip(), "steviloPrikazov": "3"},
            timeout=8,
            headers={"Accept": "application/json"},
        )
        if resp.status_code != 200:
            return {}
        data = resp.json()
        # AJPES may return list directly or wrapped in a key
        items = data if isinstance(data, list) else (
            data.get("items") or data.get("results") or data.get("data") or []
        )
        if not items:
            return {}
        first = items[0]
        subj_id = first.get("subjektMRSId") or first.get("id") or ""
        if not subj_id:
            return _ajpes_normalize(first)
        # Fetch full details
        detail = requests.get(
            AJPES_PRS_DETAIL,
            params={"format": "json", "subjektMRSId": str(subj_id)},
            timeout=8,
        )
        if detail.status_code == 200:
            try:
                return _ajpes_normalize(detail.json())
            except Exception:
                pass
        return _ajpes_normalize(first)
    except Exception:
        return {}


def _detect_high_risk_jurisdiction(text: str) -> list[str]:
    """Return list of high-risk jurisdiction keywords found in text."""
    lower = text.lower()
    return [j for j in HIGH_RISK_JURISDICTIONS if j in lower]


def llm_kyc_aml_assessment(
    entities: dict,
    email_text: str,
    analysis: dict,
    ajpes_data: dict,          # {entity_name: ajpes_dict}
    tip_stranke: str = "pravna",
) -> dict:
    """
    Full structured KYC + AML risk assessment using LLM.
    ajpes_data: dict of entity_name → ajpes_lookup() result (may be empty per entity)
    Returns kyc_entitete, aml_indikatorji, skupno_tveganje, skupna_razlaga.
    """
    # Build entity context block
    entity_lines = []
    all_entities = (
        [("stranka", e) for e in entities.get("stranke", [])] +
        [("nasprotna stran", e) for e in entities.get("nasprotna_stran", [])] +
        [("ostali", e) for e in entities.get("ostali_upleteni", [])]
    )
    for role, ename in all_entities:
        aj = ajpes_data.get(ename, {})
        if aj:
            zastopniki = ", ".join(str(z) for z in aj.get("zastopniki", [])[:4]) or "—"
            lastniki = ", ".join(str(o) for o in aj.get("lastniki", [])[:4]) or "—"
            entity_lines.append(
                f"- {ename} [{role}] | AJPES: maticna={aj.get('maticna','?')}, "
                f"status={aj.get('status','?')}, dejavnost={aj.get('dejavnost','?')}, "
                f"pravna oblika={aj.get('pravnaoblika','?')}, kapital={aj.get('kapital','?')}, "
                f"zastopniki=[{zastopniki}], lastniki=[{lastniki}]"
            )
        else:
            entity_lines.append(f"- {ename} [{role}] | AJPES: ni podatkov")
    entities_block = "\n".join(entity_lines) if entity_lines else "— ni entitet —"

    # Jurisdiction hints from email
    hrj = _detect_high_risk_jurisdiction(email_text)
    jurisdiction_hint = f"Visokorizične jurisdikcije zaznane v mailu: {', '.join(hrj)}" if hrj else "Ni zaznanih visokorizičnih jurisdikcij."

    system = """Si compliance strokovnjak v odvetniški pisarni. Oceni KYC in AML tveganje na podlagi posredovanih podatkov.

Vrni SAMO veljavni JSON v tej obliki:
{
  "kyc_entitete": [
    {
      "ime": "<ime>",
      "tip": "<fizična|pravna>",
      "pep": <true|false — politično izpostavljena oseba ali njen bližnji>,
      "pep_razlaga": "<zakaj PEP ali zakaj ne — konkretno>",
      "visoko_tvegana_jurisdikcija": <true|false>,
      "lastniška_struktura": "<ocena strukture: pregledna|nepregledna|ni podatkov>",
      "opombe": "<morebitne posebnosti>",
      "tveganje": "<nizko|srednje|visoko>"
    }
  ],
  "aml_indikatorji": [
    {
      "tip": "<proof_of_funds|lastniška_struktura|jurisdikcija|gotovinska_intenzivnost|anonimnost|nujnost|corporate_structure>",
      "prisoten": <true|false>,
      "opis": "<kaj konkretno kaže na ta indikator — ali zakaj ni tveganja>",
      "tveganje": "<nizko|srednje|visoko>"
    }
  ],
  "skupno_tveganje": "<nizko|srednje|visoko>",
  "skupna_razlaga": "<2–3 stavki: skupna ocena tveganja z najpomembnejšimi razlogi>"
}

Pravila:
- PEP = politik, javni funkcionar, vodilni v državnem podjetju, ali njihov bližnji sorodnik/poslovni partner
- Visoko tveganje AML: vsaj 2 visokorizična indikatorja ALI 1 kritičen (UAE/offshore lastnik, gotovina, anonimnost)
- Srednje tveganje: 1 indikator ali nepregledna struktura
- Vse 7 AML indikatorjev mora biti ocenjenih (prisoten: true/false)"""

    user = f"""ENTITETE IN AJPES PODATKI:
{entities_block}

TIP STRANKE (iz maila): {tip_stranke}
PRAVNO PODROČJE: {analysis.get('legal_field', '—')}
POVZETEK ZADEVE: {analysis.get('povzetek', '—')}
{jurisdiction_hint}

RELEVANTNI DELI EMAILA:
{email_text[:3000]}"""

    resp = openai_client.chat.completions.create(
        model="gpt-5.4",
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        response_format={"type": "json_object"},
        temperature=0.1,
    )
    result = json.loads(resp.choices[0].message.content)
    # Ensure skupno_tveganje is propagated back to top-level analysis fields
    result.setdefault("skupno_tveganje", "nizko")
    result.setdefault("skupna_razlaga", "")
    result.setdefault("kyc_entitete", [])
    result.setdefault("aml_indikatorji", [])
    return result


# ── Tax-Fin-Lex integration ───────────────────────────────────────────────────

def _tfl_context_block(analysis: dict, ajpes_data: dict | None = None, kyc_aml: dict | None = None) -> str:
    """Build a concise context prefix for TFL questions so the AI has full case context."""
    ent = analysis.get("entitete", {})
    stranke   = ", ".join(ent.get("stranke", [])) or "—"
    nasprotna = ", ".join(ent.get("nasprotna_stran", [])) or "—"
    ostali    = ", ".join(ent.get("ostali_upleteni", [])) or "—"

    rok = analysis.get("rok", {})
    rok_parts = []
    if rok.get("preteklo_dni") is not None:
        rok_parts.append(f"{rok['preteklo_dni']} dni preteklo od dogodka")
    if rok.get("cas_dni") is not None:
        rok_parts.append(f"{rok['cas_dni']} dni preostalo")
    if rok.get("opis"):
        rok_parts.append(rok["opis"])
    rok_str = "; ".join(rok_parts) if rok_parts else "ni konkretnega roka"

    hrj = _detect_high_risk_jurisdiction(
        " ".join(ent.get("stranke", []) + ent.get("nasprotna_stran", []) + ent.get("ostali_upleteni", []))
    )
    jurisd = ", ".join(hrj) if hrj else "—"

    lines = [
        "=== KONTEKST ZADEVE ===",
        f"Pravno področje: {analysis.get('legal_field', '—')}",
        f"Tip stranke: {analysis.get('tip_stranke', '—')}",
        f"Stranka: {stranke}",
        f"Nasprotna stran: {nasprotna}",
        f"Ostali vpleteni: {ostali}",
        f"Povzetek: {analysis.get('povzetek', '—')}",
        f"Zahtevnost: {analysis.get('zahtevnost', '—')}/10",
        f"Rok: {rok_str}",
        f"Visokorizične jurisdikcije: {jurisd}",
        f"AML tveganje: {analysis.get('aml_tveganje', '—')}",
    ]

    # AJPES summary — key ownership/director facts per entity
    if ajpes_data:
        ajpes_lines = []
        for name, aj in ajpes_data.items():
            if not aj:
                continue
            zast = ", ".join(str(z) for z in aj.get("zastopniki", [])[:3]) or "—"
            last = ", ".join(str(o) for o in aj.get("lastniki", [])[:3]) or "—"
            ajpes_lines.append(
                f"  {name}: dejavnost={aj.get('dejavnost','?')}, "
                f"zastopniki=[{zast}], lastniki=[{last}]"
            )
        if ajpes_lines:
            lines.append("AJPES podatki:\n" + "\n".join(ajpes_lines))

    # KYC highlights — PEP flags and entity risks
    if kyc_aml:
        pep_flags = [e["ime"] for e in kyc_aml.get("kyc_entitete", []) if e.get("pep")]
        high_risk = [e["ime"] for e in kyc_aml.get("kyc_entitete", []) if e.get("tveganje") == "visoko"]
        active_aml = [i["tip"] for i in kyc_aml.get("aml_indikatorji", []) if i.get("prisoten")]
        if pep_flags:
            lines.append(f"PEP osebe: {', '.join(pep_flags)}")
        if high_risk:
            lines.append(f"Visokorizične entitete: {', '.join(high_risk)}")
        if active_aml:
            lines.append(f"Aktivni AML indikatorji: {', '.join(active_aml)}")

    lines.append("======================")
    return "\n".join(lines)

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
                sources = [_tfl_fix_url(s) for s in (ev["data"] or [])]

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


def tfl_answer_questions(
    questions: list, legal_field: str,
    analysis: dict | None = None, ajpes_data: dict | None = None,
) -> list:
    """For each question ask TFL and return qa_blocks with real sources."""
    ctx = _tfl_context_block(analysis, ajpes_data) if analysis else ""
    qa_blocks = []
    for q in questions[:3]:  # max 3 to stay within rate limits
        prompt = (
            f"{ctx}\n\n"
            f"Vprašanje za zgornjo zadevo (področje: {legal_field}):\n{q}\n\n"
            "Odgovori konkretno glede na kontekst zadeve v 3–5 stavkih brez emotikonov "
            "in brez razdelkov s poudarjenimi naslovi."
        )
        result = tfl_ask(prompt, max_tokens=500)
        qa_blocks.append({
            "vprasanje": q,
            "odgovor": result["answer"],
            "sources": result["sources"],
        })
    return qa_blocks


def _parse_numbered_list(text: str) -> list[str]:
    """Extract items from a numbered list (1. item, 2) item, etc.)."""
    items = []
    for line in text.splitlines():
        m = re.match(r'^\s*\d+[\.\)]\s*(.+)', line.strip())
        if m:
            items.append(m.group(1).strip())
    return items


def tfl_generate_questions(
    legal_field: str, summary: str, jezik: str = 'sl',
    analysis: dict | None = None, ajpes_data: dict | None = None,
) -> list[str]:
    """Ask TFL to generate 3–5 follow-up questions for this matter."""
    lang = "v angleščini" if jezik == 'en' else "v slovenščini"
    ctx = _tfl_context_block(analysis, ajpes_data) if analysis else ""
    question = (
        f"{ctx}\n\n"
        f"Za zgornjo pravno zadevo s področja '{legal_field}' navedi 3–5 najpomembnejših "
        f"usmerjevalnih vprašanj, ki jih mora odvetnik zastaviti stranki za pridobitev vseh "
        f"bistvenih informacij. Vprašanja naj bodo konkretna glede na vpletene stranke, rok "
        f"in specifičnosti zadeve — ne generična. "
        f"Odgovori SAMO z oštevilčenim seznamom vprašanj (format: '1. Vprašanje?'), brez uvoda. "
        f"Vprašanja napiši {lang}."
    )
    result = tfl_ask(question, max_tokens=500)
    return _parse_numbered_list(result["answer"])[:5]


def tfl_generate_aml_checklist(
    legal_field: str, summary: str, entities: dict, aml_level: str, jezik: str = 'sl',
    analysis: dict | None = None, ajpes_data: dict | None = None, kyc_aml: dict | None = None,
) -> list[str]:
    """Ask TFL to generate ZPPDFT-2 documentation checklist for this matter."""
    lang = "v angleščini" if jezik == 'en' else "v slovenščini"
    ctx = _tfl_context_block(analysis, ajpes_data, kyc_aml) if analysis else ""
    question = (
        f"{ctx}\n\n"
        f"Za zgornjo pravno zadevo s področja '{legal_field}' (AML tveganje: {aml_level}) "
        f"katera dokumentacija je obvezna po ZPPDFT-2?\n"
        f"Upoštevaj konkretne entitete, lastniško strukturo (AJPES), PEP status in aktivne AML "
        f"indikatorje iz konteksta. Navedi dokumente z imeni entitet kjer relevantno.\n"
        f"Odgovori SAMO z oštevilčenim seznamom dokumentov (format: '1. Dokument'), brez uvoda. "
        f"Dokumenti naj bodo {lang}."
    )
    result = tfl_ask(question, max_tokens=500)
    return _parse_numbered_list(result["answer"])[:12]


def tfl_get_statutory_deadline(
    legal_field: str, summary: str, analysis: dict | None = None
) -> dict:
    """Ask TFL for the statutory deadline (in days) for this type of matter.
    Returns {days: int|None, basis: str, sources: []}."""
    ctx = ""
    if analysis:
        rok = analysis.get("rok", {})
        preteklo = rok.get("preteklo_dni")
        ctx_parts = [f"Pravno področje: {legal_field}"]
        if preteklo is not None:
            ctx_parts.append(f"Od sprožilnega dogodka je preteklo že {preteklo} dni.")
        if rok.get("opis"):
            ctx_parts.append(f"Kontekst roka: {rok['opis']}")
        ent = analysis.get("entitete", {})
        jurisd = _detect_high_risk_jurisdiction(
            " ".join(ent.get("stranke", []) + ent.get("nasprotna_stran", []))
        )
        if jurisd:
            ctx_parts.append(f"Vpletene jurisdikcije: {', '.join(jurisd)}")
        ctx = "\n".join(ctx_parts) + "\n\n"
    question = (
        f"{ctx}Za pravno zadevo s področja '{legal_field}': {summary}\n\n"
        "Kateri je najkrajši obvezni zakonski rok, ki ga določa slovensko pravo za tovrstno zadevo "
        "(npr. od sprožilnega dogodka do izpolnitve obveznosti)?\n"
        "Odgovori OBVEZNO v tem formatu in samo tako:\n"
        "ROK: [število] dni | OSNOVA: [zakon in člen]\n"
        "Primer: ROK: 72 dni | OSNOVA: GDPR čl. 33\n"
        "Če zakonskega roka ni: ROK: ni | OSNOVA: -"
    )
    result = tfl_ask(question, max_tokens=200)
    answer = result["answer"]
    m = re.search(r'ROK:\s*(\d+)\s*dni', answer, re.IGNORECASE)
    basis_m = re.search(r'OSNOVA:\s*(.+)', answer, re.IGNORECASE)
    return {
        "days": int(m.group(1)) if m else None,
        "basis": basis_m.group(1).strip() if basis_m else "",
        "sources": result["sources"][:2],
    }


_TFL_URL_MAP = {
    "legislation": "/Zakonodaja/Podrobnosti/{}",
    "court":       "/SodnaPraksa/Podrobnosti/{}",
    "publication": "/Publikacije/Podrobnosti/{}",
}


def _tfl_fix_url(item: dict) -> dict:
    """Rewrite API-path URLs and attach is_valid flag."""
    entity_id = item.get("id") or item.get("entityId", "")
    item_type = item.get("type", "")
    if entity_id and item_type in _TFL_URL_MAP:
        item["url"] = _TFL_URL_MAP[item_type].format(entity_id)
    item["is_valid"] = _tfl_is_valid(item)
    return item


def tfl_get_deadline_info(legal_field: str, summary: str) -> dict:
    """Ask TFL for statutory deadlines relevant to this matter."""
    if not TFL_API_KEY:
        return {}
    question = (
        f"Za pravno zadevo s področja '{legal_field}': {summary}\n\n"
        "Kateri zakoni določajo obvezne zakonske roke (v urah ali dneh)? "
        "Navedi samo konkretne roke z zakonsko podlago (člen, zakon). "
        "Odgovori v 3–5 stavkih brez emotikonov."
    )
    result = tfl_ask(question, max_tokens=350)
    if not result["answer"]:
        return {}
    return {
        "tfl_deadline_info": result["answer"],
        "tfl_deadline_sources": result["sources"][:3],
    }


def _tfl_is_valid(item: dict) -> bool:
    """Return True if a TFL item appears to be currently valid legislation."""
    # Check explicit validity/status fields TFL may return
    status = (item.get("status") or item.get("veljavnost") or "").lower()
    if status and any(s in status for s in ("neveljavni", "prenehal", "archived", "expired", "razveljavljen")):
        return False
    # Check validTo date field
    valid_to = item.get("validTo") or item.get("veljavnoDo") or item.get("datumPrenehanja") or ""
    if valid_to:
        try:
            if valid_to[:10] < date.today().isoformat():
                return False
        except Exception:
            pass
    return True


def tfl_search(query: str, types: list | None = None) -> list:
    """Semantic search over TFL legal database — only currently valid documents."""
    if not TFL_API_KEY:
        return []
    try:
        payload = {"query": query, "veljavnost": "veljavni"}  # ask TFL for valid-only
        if types:
            payload["types"] = types
        resp = requests.post(
            f"{TFL_BASE}/search",
            headers=TFL_HEADERS,
            json=payload,
            timeout=20,
        )
        data = resp.json()
        # Fetch more items than needed so we can filter
        items = data.get("data", {}).get("items", [])[:10]
        # Filter client-side as well (in case TFL ignores the veljavnost param)
        valid = [i for i in items if _tfl_is_valid(i)]
        return [_tfl_fix_url(i) for i in valid[:5]]
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

    jezik = analysis.get("jezik", "sl")
    legal_field = analysis.get("legal_field", "")
    povzetek = analysis.get("povzetek", "")
    entitete = analysis.get("entitete", {})
    tip_stranke = analysis.get("tip_stranke", "pravna")

    # ── AJPES lookups (parallel, non-blocking) ────────────────────────────────
    # Collect all legal-entity candidates (skip natural-person-like names)
    all_entity_names = list({
        e for group in entitete.values() for e in group if e and len(e) > 2
    })
    ajpes_data: dict = {}
    if all_entity_names:
        with ThreadPoolExecutor(max_workers=4) as pool:
            futures = {pool.submit(ajpes_lookup, name): name for name in all_entity_names[:6]}
            for fut, name in futures.items():
                result = fut.result()
                if result:
                    ajpes_data[name] = result

    # ── TFL + KYC/AML parallel group ─────────────────────────────────────────
    tfl_questions: list[str] = []
    tfl_aml_checklist: list[str] = []
    tfl_refs: list = []
    tfl_statutory: dict = {}
    kyc_aml: dict = {}

    with ThreadPoolExecutor(max_workers=6) as pool:
        fut_kyc = pool.submit(
            llm_kyc_aml_assessment, entitete, full_email, analysis, ajpes_data, tip_stranke
        )
        if TFL_API_KEY:
            # These run in parallel with KYC/AML — pass analysis+ajpes for context
            fut_q   = pool.submit(tfl_generate_questions, legal_field, povzetek, jezik, analysis, ajpes_data)
            fut_ref = pool.submit(tfl_search, povzetek, ["act", "court"])
            fut_rok = pool.submit(tfl_get_statutory_deadline, legal_field, povzetek, analysis)

        kyc_aml = fut_kyc.result()
        aml_level = kyc_aml.get("skupno_tveganje", "nizko")

        if TFL_API_KEY:
            tfl_questions = fut_q.result()
            tfl_refs      = fut_ref.result()
            tfl_statutory = fut_rok.result()
            # AML checklist runs after KYC/AML — passes full context including kyc_aml
            tfl_aml_checklist = tfl_generate_aml_checklist(
                legal_field, povzetek, entitete, aml_level, jezik,
                analysis, ajpes_data, kyc_aml,
            )

    # Compute effective deadline: min(statutory_remaining, client_days)
    rok_raw = analysis.get("rok", {})
    statutory_days  = tfl_statutory.get("days")       # total legal period
    preteklo        = rok_raw.get("preteklo_dni")      # days already elapsed
    stranka_days    = rok_raw.get("stranka_cas_dni")   # what client said
    zakonski_preostali = (statutory_days - preteklo) if (statutory_days and preteklo is not None) else statutory_days
    candidates = [d for d in [zakonski_preostali, stranka_days] if d is not None and d > 0]
    cas_dni = min(candidates) if candidates else None
    analysis["rok"] = {
        **rok_raw,
        "cas_dni": cas_dni,
        "zakonski_limit_dni": statutory_days,
        "zakonski_osnova": tfl_statutory.get("basis", ""),
        "zakonski_preostali": zakonski_preostali,
    }
    if cas_dni is not None:
        analysis["rok"]["nujno"] = True

    # Store results — keep aml_tveganje at top level for backwards compat
    analysis["aml_tveganje"] = aml_level
    analysis["aml_razlaga"]  = kyc_aml.get("skupna_razlaga", "")
    analysis["kyc_aml"]      = kyc_aml
    analysis["ajpes_data"]   = ajpes_data
    analysis["dodatna_vprasanja"] = tfl_questions
    analysis["aml_checklist"] = tfl_aml_checklist

    # Call 3 — generate draft (GPT-4o) — uses updated analysis with TFL questions
    draft_result = llm_generate_draft(full_email, analysis, selected_cases)

    # Call 4 — answer each TFL-generated question via TFL (full context available now)
    if TFL_API_KEY and tfl_questions:
        tfl_qa = tfl_answer_questions(tfl_questions, legal_field, analysis, ajpes_data)
    else:
        tfl_qa = []

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


@app.delete("/api/emails/{email_id}")
def delete_email(email_id: str):
    emails = get_emails()
    emails = [e for e in emails if e.get("id") != email_id]
    save_emails(emails)
    return {"ok": True}


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


@app.get("/api/calendar")
def get_calendar():
    """Return mock calendar availability for the next 90 days."""
    cal_file = DB / "calendar.json"
    if not cal_file.exists():
        # Auto-generate if missing
        import random as _r; _r.seed(42)
        cal = {}
        for i in range(90):
            d = (date.today() + timedelta(days=i)).isoformat()
            dow = date.fromisoformat(d).weekday()
            cal[d] = 0 if dow >= 5 else _r.choices([0,1,2,3], weights=[5,20,40,35])[0]
        cal_file.write_text(json.dumps(cal, indent=2), encoding="utf-8")
    return json.loads(cal_file.read_text(encoding="utf-8"))


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
