/** Shared helpers for turning raw similarity scores into plain language. */

/** Plain-language read of the hybrid score, for someone who isn't a data scientist.
 * Thresholds are just for display grouping - not a claim of validated accuracy.
 * Fallback only: used when no reliability calibration is available for this
 * result. Prefer reliabilityLevel() whenever `reliability` is present, so the
 * badge and the "~X% historically accurate" figure can never disagree. */
export function similarityLevel(score) {
  if (score >= 0.45) return { text: "High similarity", tone: "green" };
  if (score >= 0.28) return { text: "Moderate similarity", tone: "blue" };
  if (score >= 0.15) return { text: "Some overlap", tone: "amber" };
  return { text: "Low similarity", tone: "gray" };
}

/** Same wording/tones as similarityLevel, but driven by the calibrated
 * reliability (fraction of similarly-scored validation pairs a legal expert
 * actually judged similar) instead of an arbitrary cut on the raw score -
 * so the badge always matches the "~X% historically accurate" figure shown
 * next to it, rather than the two potentially contradicting each other. */
export function reliabilityLevel(reliability) {
  if (reliability >= 0.7) return { text: "High similarity", tone: "green" };
  if (reliability >= 0.4) return { text: "Moderate similarity", tone: "blue" };
  if (reliability >= 0.15) return { text: "Some overlap", tone: "amber" };
  return { text: "Low similarity", tone: "gray" };
}

/** Pick reliabilityLevel() when a calibration is available for this result,
 * else fall back to the raw-score similarityLevel(). Use this everywhere a
 * similarity badge is shown next to a reliability figure. */
export function bestSimilarityLevel(hybridScore, reliability) {
  return reliability != null ? reliabilityLevel(reliability) : similarityLevel(hybridScore);
}

/** Plain-language reliability caption + a tooltip explaining exactly what it
 * means, from the same calibration used for the badge. Below 15%, spell out
 * "weak match" in words instead of leaving a bare "~2%" that could read as
 * broken rather than as a genuine, honestly-reported weak result. */
export function reliabilityCaption(reliability, n) {
  if (reliability == null) return null;
  const pct = Math.round(reliability * 100);
  const weak = reliability < 0.15;
  return {
    text: weak ? "⚠ Weak match — historically almost never judged similar" : `~${pct}% historically accurate at this score`,
    weak,
    title: `Of the ${n} expert-scored validation pairs the model scored near this level, about ${pct}% were actually judged similar by legal experts. Based on a small sample (n=${n}) — a guide, not a guarantee.`,
  };
}

/** "negotiable_instruments_act_1881" -> "Negotiable Instruments Act 1881" */
export function readableAct(actId) {
  return actId
    .replace(/_/g, " ")
    .replace(/\b\w/g, (c) => c.toUpperCase())
    .replace(/\bNi\b/, "NI");
}

function wordingOverlapPhrase(textScore) {
  if (textScore >= 0.45) return "very similar - the two judgments use much of the same language";
  if (textScore >= 0.28) return "moderately similar - there's a real but partial overlap in language";
  if (textScore >= 0.12) return "only somewhat similar - the overlap in language is limited";
  return "quite different - almost none of the wording overlaps";
}

// Plain-English meaning for the handful of sections that show up constantly in
// this corpus but often arrive as "bare nodes" (no title from the curated
// hierarchy). Anything not listed here falls back to a generic phrase - this
// is a convenience gloss, not a substitute for reading the actual provision.
const KNOWN_SECTIONS = {
  "ni_act|138": "dishonour of a cheque for insufficiency of funds",
  "ni_act|141": "when a company's directors/officers can be held personally liable for an offence by the company",
  "ipc|302": "murder",
  "ipc|304": "culpable homicide not amounting to murder",
  "ipc|304b": "dowry death",
  "ipc|420": "cheating and dishonestly inducing delivery of property",
  "ipc|498a": "cruelty by a husband or his relatives",
  "crpc|482": "the High Court's inherent power to prevent abuse of process or quash proceedings",
  "constitution|226": "a High Court's power to issue writs",
  "constitution|32": "the Supreme Court's power to issue writs to enforce fundamental rights",
  "constitution|21": "the right to life and personal liberty",
  "constitution|14": "the right to equality before the law",
};

function sectionMeaning(s) {
  if (s.section_title) return s.section_title.toLowerCase();
  const key = `${(s.act_id || "").toLowerCase()}|${(s.section_number || "").toLowerCase()}`;
  return KNOWN_SECTIONS[key] || null;
}

/** Merge duplicate references to the same section (e.g. a "bare node" id
 * alongside a fully-labelled curated entry for the same provision), keeping
 * whichever copy carries the most information. */
function dedupeSections(sections) {
  // Keyed on section number alone: the same provision often shows up twice -
  // once as a "bare node" (e.g. act_id "ni_act") and once via the curated
  // hierarchy under a different act_id spelling (e.g. "negotiable_instruments_act_1881")
  // for the same Act. Merging just on the number is the practical fix; a
  // number collision across genuinely different Acts would be rare and this
  // still shows something reasonable rather than a confusing duplicate.
  const bySection = new Map();
  for (const s of sections) {
    const key = String(s.section_number);
    const existing = bySection.get(key);
    bySection.set(key, {
      ...s,
      act_id: existing?.act_id || s.act_id,
      act_name: existing?.act_name || s.act_name,
      section_title: existing?.section_title || s.section_title,
    });
  }
  return [...bySection.values()];
}

/** Build a plain-English paragraph explaining why two cases are similar -
 * leads with what the shared legal issue actually is (not just a citation),
 * then covers wording overlap and any shared case citations. */
export function plainExplanation(ex) {
  const sentences = [];
  const statutes = ex.shared_statutes || [];
  const sections = dedupeSections(statutes.filter((s) => s.kind === "section"));
  // sections with a known/curated meaning first - they make the best lead sentence
  sections.sort((a, b) => (sectionMeaning(b) ? 1 : 0) - (sectionMeaning(a) ? 1 : 0));

  if (sections.length > 0) {
    const first = sections[0];
    const actLabel = first.act_name || readableAct(first.act_id);
    const meaning = sectionMeaning(first);
    sentences.push(
      meaning
        ? `Both cases raise the same core legal issue: ${meaning}, under Section ${first.section_number} of the ${actLabel}.`
        : `Both cases involve Section ${first.section_number} of the ${actLabel}.`,
    );

    const rest = sections.slice(1);
    if (rest.length === 1) {
      const r = rest[0];
      const rMeaning = sectionMeaning(r);
      const rAct = r.act_name || readableAct(r.act_id);
      sentences.push(
        rMeaning
          ? `They also both involve Section ${r.section_number}${rAct !== actLabel ? ` of the ${rAct}` : ""} (${rMeaning}).`
          : `They also both cite Section ${r.section_number} of the ${rAct}.`,
      );
    } else if (rest.length > 1) {
      const others = rest.map((r) => `s.${r.section_number}`).join(", ");
      sentences.push(`They also share ${rest.length} more provision${rest.length === 1 ? "" : "s"} in common (${others}).`);
    }
  } else if (statutes.length > 0) {
    const acts = [...new Set(statutes.map((s) => s.act_name || readableAct(s.act_id)))];
    sentences.push(`Both cases cite ${acts.slice(0, 2).join(" and ")}${acts.length > 2 ? ", among others" : ""}, though not the same specific sections.`);
  } else {
    sentences.push("These two cases don't share any detected statute or citation - the match is based on the wording of the judgments alone.");
  }

  const terms = (ex.shared_terms || []).filter((t) => t.length > 2);
  sentences.push(`Beyond the shared law, the actual wording of the two judgments is ${wordingOverlapPhrase(ex.text_score)}.`);
  if (terms.length > 0) {
    sentences.push(`The specific words and phrases driving that are: ${terms.slice(0, 6).map((t) => `"${t}"`).join(", ")}.`);
  }

  if (ex.shared_cited_cases?.length > 0) {
    const names = ex.shared_cited_cases.slice(0, 2).map((c) => c.title || c.doc_id).join(" and ");
    sentences.push(
      `Both judgments also cite the same earlier case${ex.shared_cited_cases.length > 1 ? "s" : ""}: ${names}${ex.shared_cited_cases.length > 2 ? `, and ${ex.shared_cited_cases.length - 2} more` : ""}.`,
    );
  } else if (ex.cocitation_count > 0) {
    sentences.push(`They are also both cited together by ${ex.cocitation_count} other case${ex.cocitation_count === 1 ? "" : "s"} in the corpus.`);
  }

  sentences.push(
    "This points to a shared point of law, not necessarily identical facts - worth checking the full text before relying on it.",
  );

  return sentences.join(" ");
}

/** Turn a statute's ICF weight back into plain language. ICF is
 * ln((N+1)/(df+1)) + 1, so df (how many cases cite it) can be recovered.
 * A shared provision that few cases cite is stronger evidence of similarity
 * than one nearly every case cites (e.g. Art. 226). */
export function distinctiveness(icf, totalCases) {
  if (icf == null || !totalCases) return null;
  const df = Math.max(1, Math.round((totalCases + 1) / Math.exp(icf - 1) - 1));
  const share = df / totalCases;
  let label, tone;
  if (share < 0.005) { label = "Very distinctive"; tone = "green"; }
  else if (share < 0.03) { label = "Distinctive"; tone = "blue"; }
  else if (share < 0.15) { label = "Fairly common"; tone = "amber"; }
  else { label = "Very common"; tone = "gray"; }
  return { label, tone, df };
}
