# ============================================================
# Aspect-Based Sentiment Analysis — BIO Tagging
# ============================================================
# What is BIO Tagging?
#
#   BIO stands for:
#     B = Beginning of an aspect term
#     I = Inside (continuation) of an aspect term
#     O = Outside (not an aspect term)
#
#   Example sentence:
#     "The  battery  life  is  great  but  screen  is  bad"
#      O     B        I    O    O      O     B       O   O
#
#   Why do we need this?
#     Instead of just predicting "there is an aspect somewhere",
#     BIO tagging tells us EXACTLY which words are the aspects.
#     This is called "sequence labeling" — we label every token.
#
# Flow:
#   SemEval XML → parse → assign BIO tags → evaluate with span-F1
# ============================================================

# Install required libraries (run once in terminal):
# pip install pandas numpy matplotlib seaborn

import os
import xml.etree.ElementTree as ET
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter


# ============================================================
# STEP 1: Parse SemEval XML → sentences with aspect spans
# ============================================================

def parse_semeval_for_bio(filepath):
    """
    Reads a SemEval XML file.
    Returns a list of dicts, each with:
      - text:    the raw review sentence (string)
      - aspects: list of dicts with start, end, term, polarity
    """
    tree = ET.parse(filepath)
    root = tree.getroot()
    sentences = []

    for sentence in root.findall('.//sentence'):
        text_el = sentence.find('text')
        if text_el is None or not text_el.text:
            continue
        text = text_el.text
        aspects = []

        for container_tag in ['aspectTerms', 'Opinions']:
            container = sentence.find(container_tag)
            if container is not None:
                child_tag = 'aspectTerm' if container_tag == 'aspectTerms' else 'Opinion'
                for asp in container.findall(child_tag):
                    term     = asp.get('term', asp.get('target', ''))
                    from_idx = asp.get('from', '-1')
                    to_idx   = asp.get('to', '-1')
                    polarity = asp.get('polarity', asp.get('sentiment', 'unknown'))

                    # Skip NULL aspects (no explicit term in text)
                    if term and term.upper() != 'NULL' and from_idx != '-1':
                        aspects.append({
                            'start':    int(from_idx),
                            'end':      int(to_idx),
                            'term':     term,
                            'polarity': polarity,
                        })

        sentences.append({'text': text, 'aspects': aspects})

    return sentences


# ============================================================
# STEP 2: Tokenize and assign BIO labels
# ============================================================

def tokenize_and_bio_label(sentence_data):
    """
    Given a sentence dict (text + aspect spans in chars),
    returns (tokens, labels) — one label per token.

    Strategy:
      1. Split sentence into words
      2. Track the character position of each word
      3. For each word, check if it falls inside an aspect span
      4. First word of an aspect → B-ASP
         Continuation word      → I-ASP
         Not part of an aspect  → O
    """
    text    = sentence_data['text']
    aspects = sentence_data['aspects']

    # --- Tokenize: split on whitespace, track char positions ---
    tokens      = []
    token_spans = []   # (start_char, end_char) for each token
    pos         = 0

    for word in text.split():
        start = text.find(word, pos)
        end   = start + len(word)
        tokens.append(word)
        token_spans.append((start, end))
        pos = end

    # --- Build a character-level map: which aspect index covers each char? ---
    # char_aspect[i] = index into aspects[], or -1 if not an aspect char
    char_aspect = [-1] * len(text)
    for asp_idx, asp in enumerate(aspects):
        for ci in range(asp['start'], min(asp['end'], len(text))):
            char_aspect[ci] = asp_idx

    # --- Assign BIO labels ---
    labels          = []
    prev_aspect_idx = -1

    for tok_start, tok_end in token_spans:
        # Find if any character of this token belongs to an aspect
        asp_idx = -1
        for ci in range(tok_start, tok_end):
            if ci < len(char_aspect) and char_aspect[ci] != -1:
                asp_idx = char_aspect[ci]
                break

        if asp_idx == -1:
            labels.append('O')
            prev_aspect_idx = -1
        elif asp_idx != prev_aspect_idx:
            labels.append('B-ASP')   # First token of a new aspect
            prev_aspect_idx = asp_idx
        else:
            labels.append('I-ASP')   # Continuation of same aspect

    return tokens, labels


# ============================================================
# STEP 3: Build the full BIO dataset
# ============================================================

def build_bio_dataset(sentences):
    """
    Takes a list of parsed sentence dicts.
    Returns a list of {tokens, labels, text} dicts.
    """
    dataset = []
    for sent in sentences:
        tokens, labels = tokenize_and_bio_label(sent)
        if tokens:
            dataset.append({
                'tokens': tokens,
                'labels': labels,
                'text':   sent['text'],
            })
    return dataset


# ============================================================
# STEP 4: Demo data (used if no XML files are found)
# ============================================================

DEMO_SENTENCES = [
    {
        'text': 'The battery life is excellent but the screen is terrible.',
        'aspects': [
            {'start': 4,  'end': 16, 'term': 'battery life', 'polarity': 'positive'},
            {'start': 37, 'end': 43, 'term': 'screen',       'polarity': 'negative'},
        ]
    },
    {
        'text': 'Great food but the service was really slow.',
        'aspects': [
            {'start': 6,  'end': 10, 'term': 'food',    'polarity': 'positive'},
            {'start': 19, 'end': 26, 'term': 'service', 'polarity': 'negative'},
        ]
    },
    {
        'text': 'The pasta was average and the ambiance was wonderful.',
        'aspects': [
            {'start': 4,  'end': 9,  'term': 'pasta',    'polarity': 'neutral'},
            {'start': 30, 'end': 37, 'term': 'ambiance', 'polarity': 'positive'},
        ]
    },
    {
        'text': 'Keyboard feels mushy but the display is razor sharp.',
        'aspects': [
            {'start': 0,  'end': 8,  'term': 'Keyboard', 'polarity': 'negative'},
            {'start': 29, 'end': 36, 'term': 'display',  'polarity': 'positive'},
        ]
    },
]

# Load real data if available, otherwise use demo
DATA_FILE = 'data/Restaurants_Train_v2.xml'
if os.path.exists(DATA_FILE):
    print(f"✅ Loading real data from {DATA_FILE}")
    raw_sentences = parse_semeval_for_bio(DATA_FILE)
else:
    print("📌 No XML file found — using demo sentences.")
    print("   Place SemEval XML files in data/ for real training.\n")
    raw_sentences = DEMO_SENTENCES

bio_dataset = build_bio_dataset(raw_sentences)


# ============================================================
# STEP 5: Print BIO examples — so you can see it clearly
# ============================================================

print("=" * 55)
print("BIO TAGGING EXAMPLES")
print("=" * 55)

for i, sample in enumerate(bio_dataset[:4]):
    print(f"\n📝 Sentence {i+1}: {sample['text']}")
    print(f"  {'Token':<20} Label")
    print("  " + "-" * 35)
    for tok, lbl in zip(sample['tokens'], sample['labels']):
        if lbl == 'B-ASP':
            note = "  ← ASPECT BEGINS HERE"
        elif lbl == 'I-ASP':
            note = "  ← aspect continues"
        else:
            note = ""
        print(f"  {tok:<20} {lbl}{note}")


# ============================================================
# STEP 6: Label statistics
# ============================================================

all_labels   = [lbl for s in bio_dataset for lbl in s['labels']]
label_counts = Counter(all_labels)

b = label_counts.get('B-ASP', 0)
i = label_counts.get('I-ASP', 0)
o = label_counts.get('O', 0)

print(f"\n{'='*55}")
print("LABEL DISTRIBUTION")
print(f"{'='*55}")
print(f"  B-ASP (aspect start) : {b:>6}")
print(f"  I-ASP (aspect cont.) : {i:>6}")
print(f"  O     (non-aspect)   : {o:>6}")
print(f"  Total tokens         : {b+i+o:>6}")
print(f"\n  Imbalance ratio (O vs aspect): {o/(b+i+1):.1f} : 1")
print("  ⚠️  This imbalance is expected — most words are NOT aspects.")
print("     Use weighted loss or focal loss when training your model.")

# Average aspect term length (in tokens)
aspect_lengths = []
for sample in bio_dataset:
    in_span = False
    span_len = 0
    for lbl in sample['labels']:
        if lbl == 'B-ASP':
            if in_span:
                aspect_lengths.append(span_len)
            in_span  = True
            span_len = 1
        elif lbl == 'I-ASP' and in_span:
            span_len += 1
        else:
            if in_span:
                aspect_lengths.append(span_len)
            in_span  = False
            span_len = 0
    if in_span:
        aspect_lengths.append(span_len)

if aspect_lengths:
    print(f"\n  Avg aspect term length : {np.mean(aspect_lengths):.2f} tokens")
    print(f"  Max aspect term length : {max(aspect_lengths)} tokens")
    single = sum(1 for l in aspect_lengths if l == 1)
    multi  = sum(1 for l in aspect_lengths if l > 1)
    print(f"  Single-word aspects    : {single} ({100*single/len(aspect_lengths):.1f}%)")
    print(f"  Multi-word aspects     : {multi}  ({100*multi/len(aspect_lengths):.1f}%)")


# ============================================================
# STEP 7: Span-level F1 Score
# ============================================================
# Why span-F1 and NOT token-level accuracy?
#
#   If 90% of tokens are O, a model that always predicts O
#   gets 90% accuracy — but finds ZERO aspects. Worthless!
#
#   Span-F1 only gives credit when the ENTIRE aspect span
#   is correctly identified (start and end both match).
#
#   Precision = correct spans found / all spans predicted
#   Recall    = correct spans found / all true spans
#   F1        = harmonic mean of precision and recall (balance)

def extract_spans(labels):
    """
    From a BIO label list, return a set of (start, end) index pairs.
    Example: ['O','B-ASP','I-ASP','O'] → {(1, 2)}
    """
    spans   = set()
    start   = None
    for idx, label in enumerate(labels):
        if label == 'B-ASP':
            if start is not None:          # close previous span
                spans.add((start, idx - 1))
            start = idx
        elif label != 'I-ASP':            # O closes any open span
            if start is not None:
                spans.add((start, idx - 1))
                start = None
    if start is not None:
        spans.add((start, len(labels) - 1))
    return spans


def span_f1(true_labels_list, pred_labels_list):
    """
    Computes span-level Precision, Recall, F1.

    Args:
      true_labels_list : list of gold label sequences
      pred_labels_list : list of predicted label sequences
    Returns:
      dict with precision, recall, f1, tp, fp, fn
    """
    tp = fp = fn = 0
    for true_lbls, pred_lbls in zip(true_labels_list, pred_labels_list):
        true_spans = extract_spans(true_lbls)
        pred_spans = extract_spans(pred_lbls)
        tp += len(true_spans & pred_spans)
        fp += len(pred_spans - true_spans)
        fn += len(true_spans - pred_spans)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = (2 * precision * recall / (precision + recall)
                 if (precision + recall) > 0 else 0.0)
    return {'precision': precision, 'recall': recall, 'f1': f1,
            'tp': tp, 'fp': fp, 'fn': fn}


# --- Simulate predictions to demonstrate the F1 metric ---
# In real use, these come from your trained NER model.

true_labels_list = [s['labels'] for s in bio_dataset[:4]]

# Perfect predictions (copy of true) → F1 = 1.0
perfect_preds = [list(lbls) for lbls in true_labels_list]

# Partial predictions (miss some aspects) → lower recall
partial_preds = []
for lbls in true_labels_list:
    pred = []
    skip = False
    for lbl in lbls:
        if lbl == 'B-ASP' and np.random.random() < 0.4:
            skip = True   # randomly miss some aspects
        if skip and lbl in ('B-ASP', 'I-ASP'):
            pred.append('O')
        else:
            pred.append(lbl)
            if lbl == 'O':
                skip = False
    partial_preds.append(pred)

# Bad predictions (random guessing) → low F1
bad_preds = []
for lbls in true_labels_list:
    bad_preds.append(
        np.random.choice(['O', 'B-ASP', 'I-ASP'],
                         size=len(lbls),
                         p=[0.7, 0.2, 0.1]).tolist()
    )

scenarios = {
    'Perfect model':  span_f1(true_labels_list, perfect_preds),
    'Partial model':  span_f1(true_labels_list, partial_preds),
    'Random guessing': span_f1(true_labels_list, bad_preds),
}

print(f"\n{'='*55}")
print("SPAN-LEVEL F1 DEMO (simulated predictions)")
print(f"{'='*55}")
print(f"  {'Scenario':<20} {'Precision':>10} {'Recall':>10} {'F1':>10}")
print("  " + "-" * 52)
for name, scores in scenarios.items():
    print(f"  {name:<20} {scores['precision']:>10.3f} {scores['recall']:>10.3f} {scores['f1']:>10.3f}")


# ============================================================
# STEP 8: Visualizations
# ============================================================

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle('BIO Tagging — Analysis', fontsize=14, fontweight='bold')

# Plot 1: Label distribution pie chart
ax = axes[0]
sizes  = [b, i, o]
colors = ['#2ecc71', '#3498db', '#e0e0e0']
labels_pie = [f'B-ASP\n({b})', f'I-ASP\n({i})', f'O\n({o})']
ax.pie(sizes, labels=labels_pie, colors=colors, autopct='%1.1f%%',
       startangle=90, textprops={'fontsize': 10})
ax.set_title('Token Label Distribution')

# Plot 2: Aspect term length distribution
ax = axes[1]
if aspect_lengths:
    length_counts = Counter(aspect_lengths)
    xs = sorted(length_counts.keys())
    ys = [length_counts[x] for x in xs]
    ax.bar(xs, ys, color='#5dade2', edgecolor='black')
    ax.set_title('Aspect Term Length (tokens)')
    ax.set_xlabel('Number of tokens in aspect')
    ax.set_ylabel('Count')
    ax.set_xticks(xs)
else:
    ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)

# Plot 3: F1 comparison bar chart
ax = axes[2]
scenario_names = list(scenarios.keys())
f1_scores      = [scenarios[s]['f1'] for s in scenario_names]
prec_scores    = [scenarios[s]['precision'] for s in scenario_names]
rec_scores     = [scenarios[s]['recall'] for s in scenario_names]

x     = np.arange(len(scenario_names))
width = 0.25
ax.bar(x - width, prec_scores, width, label='Precision', color='#a29bfe', edgecolor='black')
ax.bar(x,         f1_scores,   width, label='F1',        color='#fd79a8', edgecolor='black')
ax.bar(x + width, rec_scores,  width, label='Recall',    color='#55efc4', edgecolor='black')
ax.set_title('Span-F1 by Model Quality')
ax.set_xticks(x)
ax.set_xticklabels([n.replace(' ', '\n') for n in scenario_names], fontsize=8)
ax.set_ylabel('Score')
ax.set_ylim(0, 1.1)
ax.legend(fontsize=8)
ax.axhline(y=0.8, color='gray', linestyle='--', alpha=0.5, label='Good threshold (0.8)')

plt.tight_layout()
plt.savefig('bio_tagging_analysis.png', dpi=150, bbox_inches='tight')
plt.show()
print("\n✅ Plot saved to 'bio_tagging_analysis.png'")


# ============================================================
# STEP 9: Export BIO-tagged data to CSV (for model training)
# ============================================================
# Models like BERT-NER need the data in token-per-row format.
# We also add a sentence_id column to group tokens.

rows = []
for sent_id, sample in enumerate(bio_dataset):
    for tok, lbl in zip(sample['tokens'], sample['labels']):
        rows.append({
            'sentence_id': sent_id,
            'token':       tok,
            'label':       lbl,
        })

bio_df = pd.DataFrame(rows)
bio_df.to_csv('bio_tagged_data.csv', index=False)
print(f"✅ BIO-tagged dataset saved to 'bio_tagged_data.csv' ({len(bio_df)} rows)")
print("\n🎉 BIO Tagging complete!")
print("   Next step: Fine-tune BERT for aspect sentiment classification.")
