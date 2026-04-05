# ============================================================
# ABSA — Rule-Based Baseline using spaCy Dependency Parsing
# ============================================================
# Why do we need a baseline?
#
#   Before claiming BERT is good, we need to ask:
#   "Good compared to what?"
#
#   A rule-based baseline uses hand-crafted linguistic rules
#   (no learning, no training data) to extract aspects and
#   assign sentiment. It's simple but surprisingly decent.
#
#   If BERT only beats this by 2%, it's not very impressive.
#   If BERT beats it by 20%, that justifies the complexity.
#
# Approach:
#   1. Parse each sentence with spaCy (POS tags + dependencies)
#   2. Extract nouns/noun phrases as candidate aspects
#   3. Find sentiment words linked to each aspect via dependency tree
#   4. Classify sentiment using a lexicon (SentiWordNet / custom)
#
# ============================================================
# Install:
#   pip install spacy textblob
#   python -m spacy download en_core_web_sm
# ============================================================

import os
import xml.etree.ElementTree as ET
import pandas as pd
import numpy as np
import spacy
import re
from collections import defaultdict
from sklearn.metrics import classification_report, f1_score, precision_score, recall_score
import matplotlib.pyplot as plt
import seaborn as sns

# Load spaCy
print("Loading spaCy model...")
nlp = spacy.load('en_core_web_sm')
print("✅ spaCy loaded\n")


# ============================================================
# STEP 1: Sentiment Lexicon
# ============================================================
# A simple positive/negative word list.
# In real use, you'd use SentiWordNet or VADER.

POSITIVE_WORDS = {
    'good', 'great', 'excellent', 'amazing', 'wonderful', 'fantastic',
    'outstanding', 'superb', 'brilliant', 'perfect', 'delicious', 'loved',
    'enjoy', 'enjoyed', 'love', 'best', 'awesome', 'incredible', 'impressive',
    'beautiful', 'nice', 'pleasant', 'friendly', 'helpful', 'fresh', 'clean',
    'fast', 'quick', 'efficient', 'comfortable', 'cozy', 'warm', 'sharp',
    'vivid', 'solid', 'premium', 'generous', 'divine', 'stunning', 'rich',
    'responsive', 'lightweight', 'durable', 'reliable', 'outstanding',
    'attentive', 'spotless', 'smooth', 'crisp', 'spacious', 'tasty'
}

NEGATIVE_WORDS = {
    'bad', 'terrible', 'awful', 'horrible', 'poor', 'worst', 'disgusting',
    'disappointing', 'unacceptable', 'slow', 'rude', 'cold', 'dirty',
    'broken', 'useless', 'mediocre', 'average', 'bland', 'boring',
    'overpriced', 'expensive', 'cheap', 'noisy', 'loud', 'mushy',
    'uncomfortable', 'limited', 'weak', 'dull', 'stale', 'flat',
    'unresponsive', 'distracting', 'annoying', 'painful', 'shockingly',
    'disinterested', 'ignored', 'neglected', 'lacking', 'greasy', 'soggy'
}

NEGATION_WORDS = {'not', "n't", 'never', 'no', 'neither', 'nor', 'hardly', 'barely'}

NEUTRAL_INDICATORS = {'okay', 'ok', 'fine', 'decent', 'average', 'standard',
                      'typical', 'normal', 'ordinary', 'adequate', 'acceptable'}


def lookup_sentiment(word, negated=False):
    """
    Returns 'positive', 'negative', or 'neutral' for a word.
    Handles negation (not good → negative).
    """
    w = word.lower().strip()

    if w in NEUTRAL_INDICATORS:
        return 'neutral'

    if w in POSITIVE_WORDS:
        return 'negative' if negated else 'positive'

    if w in NEGATIVE_WORDS:
        return 'positive' if negated else 'negative'

    return None   # unknown word


# ============================================================
# STEP 2: Aspect Extraction using Dependency Parsing
# ============================================================
# spaCy gives us a dependency tree for each sentence.
# Key dependency relations we use:
#
#   nsubj  = nominal subject  (subject of a verb)
#   dobj   = direct object
#   nmod   = noun modifier
#   amod   = adjectival modifier (adjective modifying a noun)
#   advmod = adverbial modifier
#
# Pattern: NOUN ←[amod]— ADJECTIVE
#   "The food was delicious" → food (noun) ← delicious (amod)
#   "Terrible service"       → service (noun) ← terrible (amod)

def extract_aspects_with_sentiment(text):
    """
    Uses dependency parsing to extract (aspect, sentiment) pairs.

    Strategy:
      1. Find all nouns in the sentence
      2. For each noun, look for adjectives linked via:
         - amod  (direct modifier: "great food")
         - attr  (predicate: "the food was great")
         - advmod (adverb: "incredibly fast service")
      3. Check for negation on the modifier
      4. Classify sentiment from the modifier word
    """
    doc = spacy.tokens.Doc
    doc = nlp(text)
    results = []
    seen_aspects = set()

    for token in doc:
        # Focus on nouns as potential aspects
        if token.pos_ not in ('NOUN', 'PROPN'):
            continue

        aspect_term = token.text
        if aspect_term.lower() in ('i', 'we', 'you', 'it', 'this', 'that'):
            continue

        # Check for noun compound (e.g., "battery life" → take full phrase)
        noun_phrase = ' '.join(
            t.text for t in token.subtree
            if t.dep_ in ('compound', 'nmod') or t == token
        ).strip()

        # --- Find linked sentiment words ---
        sentiment_words = []
        negated = False

        for child in token.children:
            # Direct adjectival modifier: "great food", "slow service"
            if child.dep_ in ('amod', 'advmod'):
                # Check if this modifier is negated
                neg = any(c.dep_ == 'neg' or c.text.lower() in NEGATION_WORDS
                          for c in child.children)
                sent = lookup_sentiment(child.text, negated=neg)
                if sent:
                    sentiment_words.append(sent)

        # Check predicate adjective: "The food was great"
        # token ←nsubj— verb —attr→ adjective
        if token.dep_ == 'nsubj' and token.head.pos_ == 'VERB':
            verb = token.head
            neg  = any(c.dep_ == 'neg' or c.text.lower() in NEGATION_WORDS
                       for c in verb.children)
            for sibling in verb.children:
                if sibling.dep_ in ('attr', 'acomp', 'xcomp'):
                    sent = lookup_sentiment(sibling.text, negated=neg)
                    if sent:
                        sentiment_words.append(sent)
                    # Also check adverb modifiers of the adjective
                    for grandchild in sibling.children:
                        if grandchild.dep_ == 'advmod':
                            sent2 = lookup_sentiment(grandchild.text)
                            if sent2:
                                sentiment_words.append(sent2)

        # Determine final sentiment from collected words
        if sentiment_words:
            pos = sentiment_words.count('positive')
            neg = sentiment_words.count('negative')
            neu = sentiment_words.count('neutral')

            if pos > neg and pos >= neu:
                final_sentiment = 'positive'
            elif neg > pos and neg >= neu:
                final_sentiment = 'negative'
            else:
                final_sentiment = 'neutral'

            key = aspect_term.lower()
            if key not in seen_aspects:
                results.append({
                    'aspect':    noun_phrase if noun_phrase else aspect_term,
                    'sentiment': final_sentiment,
                    'sentence':  text,
                })
                seen_aspects.add(key)

    return results


# ============================================================
# STEP 3: Load SemEval data for evaluation
# ============================================================

def parse_semeval(filepath):
    """Parse SemEval XML into list of (text, aspect, polarity) dicts."""
    tree = ET.parse(filepath)
    root = tree.getroot()
    samples = []
    for sentence in root.findall('.//sentence'):
        text_el = sentence.find('text')
        if text_el is None or not text_el.text:
            continue
        text = text_el.text.strip()
        for container_tag in ['aspectTerms', 'Opinions']:
            container = sentence.find(container_tag)
            if container is not None:
                child_tag = 'aspectTerm' if container_tag == 'aspectTerms' else 'Opinion'
                for asp in container.findall(child_tag):
                    term     = asp.get('term', asp.get('target', ''))
                    polarity = asp.get('polarity', asp.get('sentiment', ''))
                    if term and term.upper() != 'NULL' and polarity in ('positive', 'neutral', 'negative'):
                        samples.append({'text': text, 'aspect': term, 'polarity': polarity})
    return samples


# Demo data
DEMO_DATA = [
    {'text': 'The food was amazing and portions were generous.',   'aspect': 'food',     'polarity': 'positive'},
    {'text': 'The food was amazing and portions were generous.',   'aspect': 'portions', 'polarity': 'positive'},
    {'text': 'Service was incredibly slow and staff were rude.',   'aspect': 'service',  'polarity': 'negative'},
    {'text': 'Service was incredibly slow and staff were rude.',   'aspect': 'staff',    'polarity': 'negative'},
    {'text': 'The ambiance was cozy but pasta was just average.',  'aspect': 'ambiance', 'polarity': 'positive'},
    {'text': 'The ambiance was cozy but pasta was just average.',  'aspect': 'pasta',    'polarity': 'neutral'},
    {'text': 'Battery life is outstanding, screen is sharp.',      'aspect': 'battery life', 'polarity': 'positive'},
    {'text': 'Battery life is outstanding, screen is sharp.',      'aspect': 'screen',   'polarity': 'positive'},
    {'text': 'Keyboard feels mushy and fan noise is very loud.',   'aspect': 'keyboard', 'polarity': 'negative'},
    {'text': 'Keyboard feels mushy and fan noise is very loud.',   'aspect': 'fan noise','polarity': 'negative'},
    {'text': 'The dessert was divine, wine selection decent.',     'aspect': 'dessert',  'polarity': 'positive'},
    {'text': 'The dessert was divine, wine selection decent.',     'aspect': 'wine selection', 'polarity': 'neutral'},
]

DATA_FILE = 'data/Restaurants_Train_v2.xml'
if os.path.exists(DATA_FILE):
    print(f"✅ Loading real data from {DATA_FILE}")
    samples = parse_semeval(DATA_FILE)
else:
    print("📌 Using demo data.\n")
    samples = DEMO_DATA


# ============================================================
# STEP 4: Run baseline predictions and evaluate
# ============================================================

print("=" * 55)
print("RULE-BASED BASELINE — SPACY DEPENDENCY PARSING")
print("=" * 55)

true_labels = []
pred_labels = []
matched     = 0
total       = 0

for sample in samples:
    text         = sample['text']
    true_aspect  = sample['aspect'].lower()
    true_polarity = sample['polarity']
    total        += 1

    # Get baseline predictions for this sentence
    predictions = extract_aspects_with_sentiment(text)

    # Find if baseline predicted sentiment for this aspect
    pred_polarity = 'neutral'   # default if aspect not found
    for pred in predictions:
        if true_aspect in pred['aspect'].lower() or pred['aspect'].lower() in true_aspect:
            pred_polarity = pred['sentiment']
            matched += 1
            break

    true_labels.append(true_polarity)
    pred_labels.append(pred_polarity)

# --- Print results ---
print(f"\nAspect coverage: {matched}/{total} ({matched/total:.1%}) aspects found by rules")
print(f"\nClassification Report:")
print(classification_report(
    true_labels, pred_labels,
    labels=['positive', 'neutral', 'negative'],
    target_names=['positive', 'neutral', 'negative'],
    zero_division=0,
))

macro_f1 = f1_score(true_labels, pred_labels, average='macro', zero_division=0)
print(f"Macro F1: {macro_f1:.4f}")
print("\n⚠️  Note: Rule-based systems struggle with:")
print("   - Implicit aspects ('The wait was forever' → service)")
print("   - Sarcasm and irony")
print("   - Complex sentence structures")
print("   → This is WHY we need BERT fine-tuning.")


# ============================================================
# STEP 5: Show example predictions
# ============================================================

print("\n--- Example predictions ---")
test_sentences = [
    "The food was absolutely delicious but the service was terrible.",
    "Battery life is outstanding but the keyboard feels cheap.",
    "The ambiance was nice, pasta was just okay.",
]

for sent in test_sentences:
    preds = extract_aspects_with_sentiment(sent)
    print(f"\n📝 {sent}")
    if preds:
        for p in preds:
            emoji = {'positive': '🟢', 'neutral': '🟡', 'negative': '🔴'}[p['sentiment']]
            print(f"   {emoji} '{p['aspect']}' → {p['sentiment']}")
    else:
        print("   (no aspects found)")


# ============================================================
# STEP 6: Visualize results
# ============================================================

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('Rule-Based Baseline — spaCy Dependency Parsing', fontsize=13, fontweight='bold')

# Plot 1: Confusion matrix
ax = axes[0]
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(true_labels, pred_labels,
                      labels=['positive', 'neutral', 'negative'])
sns.heatmap(cm, annot=True, fmt='d', ax=ax,
            xticklabels=['positive', 'neutral', 'negative'],
            yticklabels=['positive', 'neutral', 'negative'],
            cmap='Blues')
ax.set_title('Confusion Matrix — Baseline')
ax.set_xlabel('Predicted')
ax.set_ylabel('True')

# Plot 2: Per-class F1 comparison placeholder
# (will be filled in compare_models.py with BERT scores)
ax = axes[1]
classes     = ['positive', 'neutral', 'negative']
baseline_f1 = [
    f1_score(true_labels, pred_labels, labels=[c], average='micro', zero_division=0)
    for c in classes
]
x      = np.arange(len(classes))
bars   = ax.bar(x, baseline_f1, color=['#2ecc71', '#f39c12', '#e74c3c'],
                edgecolor='black', width=0.5, label='Rule-based baseline')
ax.set_title('Per-Class F1 — Baseline\n(add BERT scores in ablation script)')
ax.set_xticks(x)
ax.set_xticklabels(classes)
ax.set_ylabel('F1 Score')
ax.set_ylim(0, 1)
ax.legend()
for bar in bars:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
            f'{bar.get_height():.2f}', ha='center', fontsize=10)

plt.tight_layout()
plt.savefig('baseline_results.png', dpi=150, bbox_inches='tight')
plt.show()
print("\n✅ Baseline results saved to 'baseline_results.png'")

# Save scores for use in ablation script
import json
scores = {
    'macro_f1':  macro_f1,
    'per_class': {c: f1_score(true_labels, pred_labels, labels=[c],
                              average='micro', zero_division=0)
                  for c in ['positive', 'neutral', 'negative']},
    'coverage':  matched / total,
}
with open('baseline_scores.json', 'w') as f:
    json.dump(scores, f, indent=2)
print("✅ Scores saved to 'baseline_scores.json' (used in ablation script)")
print("\n🎉 Rule-based baseline complete!")
