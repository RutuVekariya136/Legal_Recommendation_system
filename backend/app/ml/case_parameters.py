"""Generalized case parameters, per case type, with manual entry support.

Every case has details that apply regardless of type (parties, date, place,
statutes cited ...) plus details specific to its type (weapon / post-mortem for
a criminal case, cheque number / demand notice for a cheque-bounce case ...).
This module defines a fixed parameter set per category (matching
``case_category.CATEGORIES``) and can:

  * ``detect``      - which parameters does an uploaded text already state
                      (keyword/pattern match, with the matching snippet);
  * ``augment_text``- add details the user typed in to the case text, so the
                      normal similarity pipeline sees them;
  * ``match_details``- for a candidate case, which of the entered details does
                      it also mention.

Detection is a heuristic pattern matcher, not an NLP model: "found" shows the
matching snippet so it can be checked; "missing" only means no pattern matched.

No FastAPI / DB dependency.
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass
from typing import Optional

_MONTHS = (r"(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|june?|july?|aug(?:ust)?|"
           r"sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)")


@dataclass(frozen=True)
class ParamSpec:
    id: str
    label: str
    pattern: re.Pattern
    why: str
    placeholder: str
    ftype: str = "text"          # text | date | number | select
    options: tuple = ()          # allowed values for select


def _p(pid: str, label: str, regex: str, why: str, placeholder: str,
       ftype: str = "text", options: tuple = ()) -> ParamSpec:
    return ParamSpec(pid, label, re.compile(regex, re.IGNORECASE), why, placeholder, ftype, options)


WEAPONS = ("Firearm", "Knife / blade", "Blunt object", "Axe", "Poison", "Acid", "Fire", "Vehicle", "None / bare hands", "Other")
RELATIONSHIPS = ("Spouse / partner", "Family member", "Neighbour", "Employer / employee", "Business partner",
                 "Acquaintance", "Stranger", "Other")


COMMON: list[ParamSpec] = [
    _p("parties", "Parties", r"\b(?:appellant|respondent|petitioner|complainant|accused|plaintiff|defendant|versus|vs?\.)\b",
       "Who is on each side lets cases be matched on the same party roles.", "e.g. State vs. individual accused"),
    _p("date", "Date of event", rf"\b\d{{1,2}}[./-]\d{{1,2}}[./-]\d{{2,4}}\b|\b\d{{1,2}}(?:st|nd|rd|th)?\s+(?:of\s+)?{_MONTHS}\.?,?\s+\d{{4}}\b|\b{_MONTHS}\.?\s+\d{{1,2}},?\s+\d{{4}}\b",
       "Dates fix the sequence of events and limitation periods.", "e.g. 11 June 2019", "date"),
    _p("place", "Place", r"\b(?:village|town|city|district|police station|tehsil|taluk[a]?|mohalla|colony|high court|sessions court|magistrate|tribunal)\b",
       "Place or forum identifies jurisdiction.", "e.g. village Rampur, district Indore"),
    _p("statutes", "Section / Act cited", r"\b(?:section|sec\.|s\.|article)\s*\d+[a-z]?\b|\bact,?\s+(?:of\s+)?\d{4}\b",
       "Cited sections feed the statute-network signal directly; without any, matching relies on wording alone.", "e.g. Section 302 IPC, Section 27 Evidence Act"),
    _p("citations", "Earlier case cited", r"\b[A-Z][\w.&'-]+(?:\s+[A-Z][\w.&'-]+){0,3}\s+(?:v\.|vs\.?|versus)\s+[A-Z][\w.&'-]+|\bAIR\s+\d{4}\b|\bSCC\b|\bSCR\b",
       "Cited precedents feed the citation-network signal.", "e.g. Hanumant v. State of Madhya Pradesh"),
    _p("relief", "Relief sought", r"\b(?:prayer|prays|relief|seeks?|sought|claims?|claimed|decree|compensation|set aside|quash(?:ed|ing)?)\b",
       "What outcome is sought separates otherwise similar disputes.", "e.g. set aside conviction / specific performance"),
    _p("amount", "Amount involved", r"(?:\brs\.?|\binr\b|₹)\s?[\d,]+|\b\d[\d,]*\s?(?:lakh|crore)s?\b",
       "Monetary value helps match cases of comparable scale.", "e.g. Rs. 12,00,000", "number"),
]

BY_CATEGORY: dict[str, list[ParamSpec]] = {
    "criminal_appeal": [
        _p("time", "Time of incident", r"\b\d{1,2}[:.]\d{2}\s?(?:a\.?m\.?|p\.?m\.?|hrs|hours)\b|\b(?:morning|evening|night|midnight|noon|dawn|dusk)\b|\bat about\b",
           "Time of day distinguishes alibi, last-seen and circumstantial-evidence cases.", "e.g. 10:30 pm at night"),
        _p("weapon", "Weapon / means used", r"\b(?:knife|pistol|gun|firearm|revolver|axe|sword|lathi|stick|iron rod|acid|poison|rope|blade|weapon|bullet|dagger)\b",
           "The weapon/means is a core fact for matching offence type and the recovery trail.", "choose a weapon", "select", WEAPONS),
        _p("injury", "Injuries / cause of death", r"\b(?:post[- ]?mortem|autopsy|injur(?:y|ies)|wounds?|cause of death|haemorrhage|hemorrhage|ligature)\b",
           "Medical findings separate homicide, suicide and accident fact patterns.", "e.g. post-mortem: head injury, haemorrhage"),
        _p("fir", "FIR / police report", r"\b(?:f\.?i\.?r\.?|first information report|charge[- ]?sheet|police report)\b",
           "FIR timeline is a frequent ground of appeal.", "e.g. FIR lodged next morning at police station"),
        _p("witness", "Witness evidence", r"\b(?:eye[- ]?witness(?:es)?|p\.?w\.?\s?\d+|witness(?:es)?|deposed|testif(?:y|ied|ies))\b",
           "Eye-witness vs. circumstantial evidence largely decides which precedents apply.", "e.g. one eye-witness, last seen together"),
        _p("motive", "Motive", r"\b(?:motive|enmity|quarrel|grudge|dispute|dowry|jealous(?:y)?)\b",
           "Motive is the other main axis of criminal precedent.", "e.g. land dispute, dowry demand"),
        _p("relationship", "Relationship between parties", r"\b(?:husband|wife|neighbour|neighbor|brother|sister|father|mother|son|daughter|friend|relative|employer|employee|known to)\b",
           "Whether the accused knew the victim changes the evidence and sentencing precedents that apply.", "choose a relationship", "select", RELATIONSHIPS),
        _p("recovery", "Recovery / forensic evidence", r"\b(?:recover(?:y|ed)|seiz(?:ed|ure)|forensic|f\.?s\.?l\.?|fingerprints?|chemical examiner|panchnama|memo)\b",
           "Recovery and forensic links are weighed against the chain of circumstances.", "e.g. weapon recovered on accused's pointing out"),
        _p("dying_decl", "Dying declaration / confession", r"\b(?:dying declaration|confession(?:al)?|section 27)\b",
           "Changes which admissibility precedents are relevant.", "e.g. dying declaration recorded by magistrate"),
    ],
    "cheque_bounce": [
        _p("cheque", "Cheque number / date", r"\bcheque\b[^.]{0,60}\b(?:no\.?|number|dated|bearing)\b|\bcheque no\b",
           "Identifies the instrument and the limitation clock under the NI Act.", "e.g. cheque no. 123456 dated 4 Feb 2017"),
        _p("dishonour", "Dishonour reason", r"\b(?:dishonou?r(?:ed)?|returned unpaid|return memo|insufficient funds|funds insufficient)\b",
           "Insufficient funds vs. account closed vs. stop-payment select different precedents.", "e.g. returned for insufficient funds"),
        _p("notice", "Legal demand notice", r"\b(?:demand notice|legal notice|statutory notice|notice of demand)\b",
           "Notice within the statutory period is a mandatory ingredient.", "e.g. demand notice sent 10 days after return memo"),
        _p("notice_period", "15-day payment period", r"\b(?:fifteen|15)\s+days\b",
           "Whether the 15-day window elapsed decides whether the offence is made out.", "e.g. no payment within 15 days"),
        _p("debt", "Underlying debt", r"\b(?:legally enforceable debt|liability|loan|friendly loan|consideration|invoice|goods supplied)\b",
           "Nature of the debt decides how the statutory presumption is treated.", "e.g. friendly loan / goods supplied"),
        _p("bank", "Bank / branch", r"\b(?:bank|branch|ifsc|drawee)\b",
           "Fixes territorial jurisdiction.", "e.g. State Bank, Indore branch"),
    ],
    "motor_accident": [
        _p("vehicle", "Vehicle number / type", r"\b[A-Z]{2}[- ]?\d{1,2}[- ]?[A-Z]{1,3}[- ]?\d{3,4}\b|\b(?:truck|bus|car|motorcycle|scooter|tempo|lorry|jeep|tractor)\b",
           "Ties liability to owner/insurer and groups cases by vehicle class.", "e.g. truck MP09 AB 1234"),
        _p("insurance", "Insurance policy", r"\b(?:insurance|insurer|policy no|third[- ]party)\b",
           "Insurer liability is the main contested issue in most claims.", "e.g. third-party policy valid on date"),
        _p("licence", "Driving licence", r"\b(?:driving licen[cs]e|licen[cs]ed driver|valid licen[cs]e)\b",
           "Licence validity is a standard insurer defence.", "e.g. driver held valid licence"),
        _p("negligence", "Negligence / rash driving", r"\b(?:negligen(?:ce|t)|rash and negligent|rash driving|contributory)\b",
           "Negligence finding decides compensation.", "e.g. rash and negligent driving"),
        _p("victim_profile", "Victim age / income", r"\b(?:aged|age of|years old|monthly income|annual income|salary|earning)\b",
           "Inputs to the multiplier method for compensation.", "e.g. aged 35, monthly income Rs. 20,000"),
        _p("disability", "Injury / disability", r"\b(?:disabilit(?:y|ies)|permanent|fracture|amputat|hospitali[sz]ed|medical expenses)\b",
           "Drives the compensation heads awarded.", "e.g. 40% permanent disability"),
    ],
    "constitutional": [
        _p("right", "Article / right invoked", r"\barticle\s*(?:14|19|21|32|226|300A)\b|\bfundamental right\b",
           "The right invoked is the main axis for constitutional precedent.", "e.g. Article 14 and Article 311"),
        _p("impugned", "Impugned order / law", r"\b(?:impugned|challenged|challenge to|order dated|notification|ordinance)\b",
           "Separates review of executive action from review of legislation.", "e.g. termination order dated 3 May 2020"),
        _p("state_action", "State / authority", r"\b(?:state of|union of india|government|authority|corporation|board|commissioner)\b",
           "Decides whether the action is amenable to writ jurisdiction.", "e.g. Union of India, Ministry of Textiles"),
        _p("remedy", "Alternative remedy", r"\b(?:alternative remedy|efficacious remedy|appeal lies|statutory remedy)\b",
           "A frequent preliminary bar in writ cases.", "e.g. no statutory appeal available"),
        _p("writ", "Writ sought", r"\b(?:mandamus|certiorari|prohibition|habeas corpus|quo warranto)\b",
           "Selects the applicable standard of review.", "e.g. certiorari to quash the order"),
    ],
    "tax": [
        _p("assessment_year", "Assessment year", r"\b(?:assessment year|a\.?y\.?|financial year|f\.?y\.?)\s*[:\-]?\s*\d{4}",
           "Fixes which law version and limitation rules apply.", "e.g. AY 2015-16"),
        _p("assessee", "Assessee", r"\bassessee\b",
           "Assessee category (company, individual, firm) matters.", "e.g. private limited company"),
        _p("authority", "Assessing authority / forum", r"\b(?:assessing officer|commissioner|tribunal|itat|cestat|appellate)\b",
           "The forum in the appeal chain determines binding precedents.", "e.g. ITAT Mumbai"),
        _p("order", "Assessment / demand order", r"\b(?:assessment order|demand notice|reassessment|notice under section|show[- ]cause)\b",
           "The order under challenge is what similar cases are compared against.", "e.g. reassessment notice under Section 148"),
        _p("income_head", "Income / tax head", r"\b(?:capital gains?|business income|house property|depreciation|deduction|exemption|turnover|excise duty|customs duty)\b",
           "Selects the statutory provisions and line of authority.", "e.g. capital gains, deduction under Section 80"),
    ],
    "property_land": [
        _p("property_desc", "Property description", r"\b(?:survey no\.?|plot no\.?|khasra|khata|schedule property|suit property|sq\.? ?(?:ft|m|yards?)|acres?|hectares?)\b",
           "Lets disputes be matched on property type and possession issues.", "e.g. plot no. 45, 2400 sq ft, Indore"),
        _p("agreement", "Agreement / deed", r"\b(?:agreement to sell|sale deed|gift deed|lease deed|mortgage deed|conveyance|power of attorney)\b",
           "Kind of instrument separates specific performance, title and tenancy cases.", "e.g. agreement to sell dated 4 Feb 2017"),
        _p("possession", "Possession", r"\b(?:possession|in occupation|tenant|licensee|eviction|vacant)\b",
           "Who is in possession drives part-performance analysis.", "e.g. buyer put in possession"),
        _p("title", "Title / ownership documents", r"\b(?:title|ownership|registered|registration|mutation|revenue record|patta)\b",
           "Title documents are decisive for ownership disputes.", "e.g. registered sale deed, mutation entry"),
        _p("consideration", "Consideration paid", r"\b(?:sale consideration|earnest money|advance|token amount|balance consideration)\b",
           "Payment history shows readiness and willingness.", "e.g. earnest money Rs. 12,00,000 of Rs. 40,00,000"),
    ],
    "contract_commercial": [
        _p("contract", "Contract / clause", r"\b(?:agreement|contract|clause|memorandum of understanding|mou|terms and conditions)\b",
           "The relevant clause is what a breach claim is matched against.", "e.g. supply agreement, clause 7 on delivery"),
        _p("breach", "Breach alleged", r"\b(?:breach|default|repudiat|non[- ]performance|failed to (?:pay|deliver|perform))\w*\b",
           "Separates damages, specific performance and termination cases.", "e.g. failed to deliver goods on time"),
        _p("arbitration", "Arbitration clause", r"\b(?:arbitration clause|arbitrator|arbitral|seat of arbitration)\b",
           "Changes jurisdiction and applicable precedents entirely.", "e.g. arbitration clause, seat at Delhi"),
        _p("damages", "Damages / quantum", r"\b(?:damages|liquidated damages|loss of profit|quantum)\b",
           "Quantum is compared with awards in similar disputes.", "e.g. liquidated damages Rs. 5 lakh"),
    ],
    "service_labour": [
        _p("employment", "Employment details", r"\b(?:appointed|joined service|date of appointment|designation|post of|employee|workman)\b",
           "Fixes the service rules and status that apply.", "e.g. Section Officer, appointed 2012"),
        _p("misconduct", "Charge / misconduct", r"\b(?:misconduct|charge[- ]?sheet|charges? against|disciplinary|inquiry officer)\b",
           "Primary axis for disciplinary-action precedent.", "e.g. alleged absence from duty"),
        _p("impugned_order", "Order challenged", r"\b(?:termination|dismissal|removal|compulsory retirement|suspension|discharge)\b",
           "Determines which protections are in play.", "e.g. termination simpliciter during probation"),
        _p("inquiry", "Inquiry / hearing", r"\b(?:departmental inquiry|domestic inquiry|opportunity of hearing|natural justice|show[- ]cause)\b",
           "Whether a fair inquiry was held is the most common ground of success.", "e.g. no departmental inquiry held"),
        _p("service_relief", "Relief sought", r"\b(?:reinstatement|back wages|continuity of service|re-?employment)\b",
           "Selects the comparable awards.", "e.g. reinstatement with back wages"),
    ],
    "family_matrimonial": [
        _p("marriage", "Marriage details", r"\b(?:married on|solemni[sz]ed|date of marriage|marriage was)\b",
           "Marriage date and duration matter in matrimonial law.", "e.g. married 14 Jan 2015"),
        _p("ground", "Ground alleged", r"\b(?:cruelty|desertion|adultery|irretrievable breakdown|dowry|domestic violence)\b",
           "The ground alleged is the main axis for matrimonial precedent.", "e.g. cruelty and desertion"),
        _p("children", "Children / custody", r"\b(?:child(?:ren)?|minor|custody|visitation|guardian)\b",
           "Child welfare changes the reasoning and precedents.", "e.g. one minor child, custody sought"),
        _p("maintenance", "Maintenance / income", r"\b(?:maintenance|alimony|income of|salary of|monthly (?:income|maintenance))\b",
           "Income figures are compared with awards in similar cases.", "e.g. monthly income Rs. 50,000"),
    ],
}

_STOP = {"the", "and", "for", "with", "was", "were", "from", "that", "this", "under", "not", "his", "her", "had"}
_SAMPLE_SIZE = 150
_coverage_cache: dict[str, dict] = {}


def specs_for(category: Optional[str]) -> list[tuple[str, ParamSpec]]:
    return [("common", s) for s in COMMON] + [("category", s) for s in BY_CATEGORY.get(category or "", [])]


def _snippet(text: str, m: re.Match, width: int = 50) -> str:
    s, e = max(0, m.start() - width), min(len(text), m.end() + width)
    return ("..." if s else "") + " ".join(text[s:e].split()) + ("..." if e < len(text) else "")


def _valid_date(v: str) -> bool:
    return bool(re.fullmatch(r"\d{1,2}[/.-]\d{1,2}[/.-]\d{4}|\d{4}-\d{2}-\d{2}", v.strip()))


def validate_details(details: Optional[dict], category: Optional[str]) -> tuple[dict[str, str], dict[str, str]]:
    """Lightly validate typed values. Empty values are always allowed and dropped.
    Returns (clean, issues): issues maps a parameter id to a short message for values
    that were dropped because they don't fit the field type."""
    specs = {s.id: s for _, s in specs_for(category)}
    clean, issues = {}, {}
    for k, v in clean_details(details, category).items():
        spec = specs[k]
        if spec.ftype == "date" and not _valid_date(v):
            issues[k] = "Use a date such as 11/06/2019."
        elif spec.ftype == "number" and not re.fullmatch(r"[\d,.\s]*\d[\d,.\s]*|\s*(?:rs\.?|₹|inr)?\s*[\d,.]+\s*(?:lakh|crore)?", v.strip(), re.IGNORECASE):
            issues[k] = "Enter an amount, e.g. 12,00,000."
        elif spec.ftype == "select" and v not in spec.options:
            issues[k] = "Choose one of the listed options."
        else:
            clean[k] = v
    return clean, issues


def detect(text: str, category: Optional[str]) -> list[dict]:
    """Per parameter: id, label, group, present, evidence, why, placeholder."""
    out = []
    for group, spec in specs_for(category):
        m = spec.pattern.search(text or "")
        out.append({
            "id": spec.id, "label": spec.label, "group": group, "present": m is not None,
            "evidence": _snippet(text, m) if m else None,
            "why": spec.why, "placeholder": spec.placeholder,
            "ftype": spec.ftype, "options": list(spec.options),
        })
    return out


def clean_details(details: Optional[dict], category: Optional[str]) -> dict[str, str]:
    """Keep only known parameter ids with non-empty text (length-capped)."""
    valid = {s.id for _, s in specs_for(category)}
    return {k: str(v).strip()[:300] for k, v in (details or {}).items()
            if k in valid and v is not None and str(v).strip()}


def augment_text(text: str, details: dict[str, str], category: Optional[str]) -> str:
    """The case text plus one plain sentence per entered detail, so the normal
    similarity pipeline (wording + statute/citation extraction) sees them."""
    if not details:
        return text
    labels = {s.id: s.label for _, s in specs_for(category)}
    extra = " ".join(f"{labels[k]}: {v}." for k, v in details.items() if k in labels)
    # joined with a plain space, NOT a blank line: the preprocessor treats everything
    # before the first blank line as a header and drops it from the wording score
    return f"{text} {extra}"


def match_details(candidate_text: str, details: dict[str, str]) -> dict[str, bool]:
    """Which entered details does a candidate case also mention? A detail
    matches if its full text appears, or at least half of its meaningful words do."""
    low = (candidate_text or "").lower()
    out = {}
    for pid, val in details.items():
        v = val.lower()
        if v in low:
            out[pid] = True
            continue
        toks = [t for t in re.findall(r"[a-z0-9]{3,}", v) if t not in _STOP]
        out[pid] = bool(toks) and sum(t in low for t in toks) * 2 >= len(toks)
    return out


def corpus_coverage(corpus, category: Optional[str]) -> dict:
    """Share of a deterministic sample of same-category corpus cases that state
    each parameter (cached) - shows whether a missing detail is a real gap."""
    key = category or ""
    if key in _coverage_cache:
        return _coverage_cache[key]
    ids = corpus.doc_ids_by_category(category) if category else list(corpus.doc_ids)
    if not ids:
        _coverage_cache[key] = {}
        return {}
    sample = random.Random(0).sample(ids, min(_SAMPLE_SIZE, len(ids)))
    specs = [s for _, s in specs_for(category)]
    counts = {s.id: 0 for s in specs}
    for doc_id in sample:
        txt = corpus.text(doc_id) or ""
        for s in specs:
            if s.pattern.search(txt):
                counts[s.id] += 1
    cov = {sid: round(c / len(sample), 3) for sid, c in counts.items()}
    cov["_n"] = len(sample)
    _coverage_cache[key] = cov
    return cov
