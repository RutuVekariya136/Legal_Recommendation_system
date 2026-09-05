"""Tests for app.ml.preprocessing (Step 4)."""

from __future__ import annotations

import pytest

from app.ml.preprocessing import (
    DEFAULT_CONFIG,
    LEGAL_BOILERPLATE_STOPWORDS,
    PreprocessConfig,
    clean_text,
    normalize_unicode,
    preprocess_text,
    split_header_body,
    split_sentences,
    tokenize,
)


# -- cleaning ---------------------------------------------------------------

def test_clean_text_is_idempotent():
    raw = "Foo   bar\r\n\n\n\nbaz-\nqux\tend"
    once = clean_text(raw)
    assert clean_text(once) == once


def test_clean_text_joins_hyphenated_linebreak():
    assert "judgment" in clean_text("judg-\nment")


def test_clean_text_collapses_whitespace_and_blank_lines():
    out = clean_text("a    b\n\n\n\nc")
    assert "a b" in out and "\n\n\n" not in out


def test_normalize_unicode_folds_quotes_and_dashes():
    assert normalize_unicode("“x” – y") == '"x" - y'


# -- header/body ----------------------------------------------------------

def test_split_header_body_on_numbered_paragraph():
    text = "Supreme Court of India\n30 March 1953\n1.  This is an appeal.\n2. Next."
    header, body = split_header_body(text)
    assert "Supreme Court of India" in header
    assert body.startswith("1.")


def test_split_header_body_falls_back_to_all_body():
    header, body = split_header_body("no markers here at all")
    assert header == "" and body == "no markers here at all"


# -- sentence segmentation ----------------------------------------------

def test_split_sentences_basic():
    assert split_sentences("First sentence. Second sentence.") == [
        "First sentence.",
        "Second sentence.",
    ]


@pytest.mark.parametrize(
    "text",
    [
        "The bench of M. Patanjali Sastri, J. heard the matter.",
        "See Rangachari v. Secretary of State for the point raised.",
        "It was filed under No. 204 of 1952 before the court.",
        "Relief was sought u/art. 226 of the Constitution here.",
    ],
)
def test_split_sentences_does_not_break_on_abbreviations(text):
    assert len(split_sentences(text)) == 1


# -- tokenisation -------------------------------------------------------

def test_tokenize_lowercases_and_drops_stopwords_and_punct():
    toks = tokenize("The Court, in ITS Judgment, HELD that...")
    assert "the" not in toks and "in" not in toks and "its" not in toks
    assert "court" in toks and "judgment" in toks and "held" in toks
    assert all(t.isalpha() for t in toks)


def test_tokenize_min_length():
    cfg = PreprocessConfig(min_token_len=4)
    assert "act" not in tokenize("the Act was applied", config=cfg)


def test_legal_boilerplate_is_opt_in():
    base = tokenize("the learned counsel for the appellant argued")
    assert "counsel" in base and "appellant" in base
    cfg = PreprocessConfig(remove_legal_boilerplate=True)
    stripped = tokenize("the learned counsel for the appellant argued", config=cfg)
    assert "counsel" not in stripped and "appellant" not in stripped
    assert "argued" in stripped


def test_stemming_is_opt_in():
    cfg = PreprocessConfig(stem=True)
    assert tokenize("appealing appeals appealed", config=cfg) == ["appeal", "appeal", "appeal"]


# -- end to end on real corpus ---------------------------------------

def test_preprocess_real_document():
    from app.utils.dataset import load_corpus

    corpus = load_corpus()
    pd = preprocess_text(corpus.text("1953_24"), doc_id="1953_24", config=DEFAULT_CONFIG)
    assert pd.stats["header_detected"] is True
    assert pd.stats["n_tokens"] > 1000
    assert pd.stats["n_sentences"] > 50
    assert len(pd.paragraphs) > 20
    # no leaked bare paragraph-number "sentences"
    assert not any(s.strip().rstrip(".").isdigit() for s in pd.sentences)
    assert LEGAL_BOILERPLATE_STOPWORDS  # sanity: the set is populated
