"""Tests for app.ml.case_parameters (manual case details) and the endpoints using it."""

from __future__ import annotations

from app.ml.case_parameters import (
    BY_CATEGORY, COMMON, augment_text, clean_details, detect, match_details, specs_for,
)

CRIMINAL = (
    "The appellant was convicted under Section 302 IPC. On 11-06-2019 at about 10:30 p.m. near "
    "the village of Rampur, the deceased was struck with an axe. The post-mortem report records "
    "injuries. The FIR was lodged at the police station; PW 2 deposed as an eye-witness."
)


def _by_id(rows):
    return {r["id"]: r for r in rows}


def test_detect_marks_present_and_missing():
    r = _by_id(detect(CRIMINAL, "criminal_appeal"))
    for pid in ("date", "place", "statutes", "time", "weapon", "injury", "fir", "witness"):
        assert r[pid]["present"], pid
    assert not r["motive"]["present"] and r["motive"]["evidence"] is None
    assert "axe" in r["weapon"]["evidence"].lower()


def test_uncategorised_gets_common_parameters_only():
    assert {r["id"] for r in detect(CRIMINAL, None)} == {s.id for s in COMMON}


def test_every_spec_has_reason_placeholder_and_unique_ids():
    for cat, specs in BY_CATEGORY.items():
        ids = [s.id for s in specs]
        assert len(ids) == len(set(ids)), cat
        assert all(s.why and s.placeholder for s in specs), cat
        assert not {s.id for s in COMMON} & set(ids), cat


def test_clean_details_drops_unknown_and_blank_and_caps_length():
    out = clean_details({"weapon": "  axe ", "bogus": "x", "motive": "   ", "time": "t" * 999}, "criminal_appeal")
    assert out["weapon"] == "axe" and "bogus" not in out and "motive" not in out
    assert len(out["time"]) == 300
    assert clean_details(None, "criminal_appeal") == {}


def test_augment_text_adds_labelled_sentences_only_when_given():
    assert augment_text("base", {}, "criminal_appeal") == "base"
    out = augment_text("base", {"weapon": "axe", "motive": "land dispute"}, "criminal_appeal")
    assert out.startswith("base") and "Weapon / means used: axe." in out and "Motive: land dispute." in out


def test_match_details_full_and_partial_word_match():
    text = "the accused attacked him with an axe near the river late at night"
    m = match_details(text, {"weapon": "axe", "time": "10:30 pm at night", "motive": "land dispute"})
    assert m["weapon"] is True
    assert m["time"] is True  # "night" is half of the meaningful words
    assert m["motive"] is False


def test_specs_for_unknown_category_falls_back_to_common():
    assert [s.id for _, s in specs_for("nonsense")] == [s.id for s in COMMON]


def test_augment_does_not_add_a_blank_line_that_would_hide_the_original_text():
    # the preprocessor drops everything before the first blank line as a "header"
    assert "\n\n" not in augment_text("one long line with no breaks", {"weapon": "axe"}, "criminal_appeal")


def test_validate_drops_bad_typed_values_and_keeps_empty_as_absent():
    from app.ml.case_parameters import validate_details
    clean, issues = validate_details({
        "date": "11/06/2019", "amount": "Rs. 12,00,000", "weapon": "Laser beam",
        "relationship": "Spouse / partner", "motive": "",
    }, "criminal_appeal")
    assert clean == {"date": "11/06/2019", "amount": "Rs. 12,00,000", "relationship": "Spouse / partner"}
    assert set(issues) == {"weapon"}  # "Laser beam" is not one of the listed options


def test_validate_rejects_malformed_date_and_non_numeric_amount():
    from app.ml.case_parameters import validate_details
    clean, issues = validate_details({"date": "soon", "amount": "a lot"}, "property_land")
    assert clean == {} and set(issues) == {"date", "amount"}


def test_select_and_typed_fields_are_described_to_the_frontend():
    rows = {r["id"]: r for r in detect("convicted", "criminal_appeal")}
    assert rows["weapon"]["ftype"] == "select" and "Firearm" in rows["weapon"]["options"]
    assert rows["date"]["ftype"] == "date"
    assert rows["relationship"]["ftype"] == "select"
