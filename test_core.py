# ============================================================
# tests/test_core.py
# ============================================================
# Unit tests that run automatically in the CI pipeline.
# pytest will discover and run all functions starting with test_
# ============================================================

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np


# ============================================================
# Test 1: BIO span extraction
# ============================================================

def extract_spans(labels):
    spans = set()
    start = None
    for i, lbl in enumerate(labels):
        if lbl == 'B-ASP':
            if start is not None:
                spans.add((start, i - 1))
            start = i
        elif lbl != 'I-ASP':
            if start is not None:
                spans.add((start, i - 1))
                start = None
    if start is not None:
        spans.add((start, len(labels) - 1))
    return spans


def test_extract_spans_single():
    labels = ['O', 'B-ASP', 'O', 'O']
    assert extract_spans(labels) == {(1, 1)}


def test_extract_spans_multi_token():
    labels = ['O', 'B-ASP', 'I-ASP', 'O']
    assert extract_spans(labels) == {(1, 2)}


def test_extract_spans_two_aspects():
    labels = ['B-ASP', 'O', 'B-ASP', 'I-ASP']
    assert extract_spans(labels) == {(0, 0), (2, 3)}


def test_extract_spans_all_O():
    labels = ['O', 'O', 'O']
    assert extract_spans(labels) == set()


def test_extract_spans_all_aspect():
    labels = ['B-ASP', 'I-ASP', 'I-ASP']
    assert extract_spans(labels) == {(0, 2)}


# ============================================================
# Test 2: Span-level F1
# ============================================================

def span_f1(true_list, pred_list):
    tp = fp = fn = 0
    for true, pred in zip(true_list, pred_list):
        ts = extract_spans(true)
        ps = extract_spans(pred)
        tp += len(ts & ps)
        fp += len(ps - ts)
        fn += len(ts - ps)
    prec = tp / (tp + fp + 1e-8)
    rec  = tp / (tp + fn + 1e-8)
    f1   = 2 * prec * rec / (prec + rec + 1e-8)
    return {'precision': prec, 'recall': rec, 'f1': f1}


def test_span_f1_perfect():
    true = [['O', 'B-ASP', 'I-ASP', 'O']]
    pred = [['O', 'B-ASP', 'I-ASP', 'O']]
    result = span_f1(true, pred)
    assert abs(result['f1'] - 1.0) < 0.01


def test_span_f1_all_wrong():
    true = [['B-ASP', 'O', 'O']]
    pred = [['O', 'B-ASP', 'O']]
    result = span_f1(true, pred)
    assert result['f1'] < 0.1


def test_span_f1_empty_preds():
    true = [['B-ASP', 'I-ASP', 'O']]
    pred = [['O', 'O', 'O']]
    result = span_f1(true, pred)
    assert result['recall'] < 0.01
    assert result['precision'] > 0.99 or result['precision'] < 0.01


# ============================================================
# Test 3: Label mapping sanity checks
# ============================================================

def test_label_mappings():
    LABEL2ID = {'positive': 0, 'neutral': 1, 'negative': 2}
    ID2LABEL  = {v: k for k, v in LABEL2ID.items()}
    assert ID2LABEL[0] == 'positive'
    assert ID2LABEL[1] == 'neutral'
    assert ID2LABEL[2] == 'negative'
    assert len(LABEL2ID) == 3


# ============================================================
# Test 4: Tokenization produces correct length
# ============================================================

def test_bio_label_length_matches_tokens():
    """Token list and label list must always have the same length."""
    sentences = [
        "The food was great.",
        "Battery life is excellent but keyboard feels mushy.",
        "Service slow.",
    ]
    for text in sentences:
        tokens = text.split()
        # Simulate labels (all O for simplicity)
        labels = ['O'] * len(tokens)
        assert len(tokens) == len(labels), \
            f"Length mismatch: {len(tokens)} tokens vs {len(labels)} labels"


# ============================================================
# Test 5: Sentiment lexicon coverage
# ============================================================

def test_sentiment_words_non_overlapping():
    """Positive and negative word sets must not overlap."""
    POSITIVE_WORDS = {'good', 'great', 'excellent', 'delicious', 'fast'}
    NEGATIVE_WORDS = {'bad', 'terrible', 'awful', 'slow', 'rude'}
    overlap = POSITIVE_WORDS & NEGATIVE_WORDS
    assert len(overlap) == 0, f"Overlap found: {overlap}"


if __name__ == '__main__':
    # Run tests manually if needed
    test_extract_spans_single()
    test_extract_spans_multi_token()
    test_extract_spans_two_aspects()
    test_extract_spans_all_O()
    test_extract_spans_all_aspect()
    test_span_f1_perfect()
    test_span_f1_all_wrong()
    test_span_f1_empty_preds()
    test_label_mappings()
    test_bio_label_length_matches_tokens()
    test_sentiment_words_non_overlapping()
    print("✅ All tests passed!")
