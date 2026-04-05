# ============================================================
# Aspect-Based Sentiment Analysis — Attention Visualization
# ============================================================
# What is Attention?
#
#   BERT uses "attention" to decide which words to focus on
#   when making a prediction. For example, when predicting
#   sentiment for the aspect "battery life" in:
#
#     "The battery life is great but the screen is bad"
#
#   BERT should attend heavily to "great" (near "battery life")
#   and less to "bad" (which relates to "screen").
#
#   Visualizing attention helps us:
#     - Understand WHY the model made a prediction
#     - Spot errors (e.g. model attending to wrong words)
#     - Write the interpretability section of your report
#
# Tools used:
#   - BertViz   : interactive attention head visualization
#   - matplotlib: static heatmaps for the PDF report
#
# ============================================================
# Install (run once):
#   pip install bertviz transformers torch matplotlib seaborn
# ============================================================

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
from transformers import BertTokenizerFast, BertForSequenceClassification

# BertViz — for interactive HTML visualizations
try:
    from bertviz import head_view, model_view
    from bertviz.transformers_neuron_view import BertModel as NeuronBertModel
    from bertviz.transformers_neuron_view import BertTokenizer as NeuronTokenizer
    BERTVIZ_AVAILABLE = True
except ImportError:
    print("⚠️  BertViz not installed. Run: pip install bertviz")
    print("   Static matplotlib heatmaps will still be generated.\n")
    BERTVIZ_AVAILABLE = False


# ============================================================
# STEP 1: Load your fine-tuned model
# ============================================================
# Change this path to wherever bert_finetune.py saved your model.

MODEL_PATH = 'bert_absa_model/best_model'
BASE_MODEL  = 'bert-base-uncased'

LABEL2ID = {'positive': 0, 'neutral': 1, 'negative': 2}
ID2LABEL  = {v: k for k, v in LABEL2ID.items()}
DEVICE    = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(f"🖥️  Device: {DEVICE}")

if os.path.exists(MODEL_PATH):
    print(f"✅ Loading fine-tuned model from {MODEL_PATH}")
    tokenizer = BertTokenizerFast.from_pretrained(MODEL_PATH)
    model     = BertForSequenceClassification.from_pretrained(
                    MODEL_PATH, output_attentions=True)
else:
    print(f"📌 Fine-tuned model not found at {MODEL_PATH}.")
    print(f"   Loading base BERT for demo (predictions won't be meaningful).\n")
    tokenizer = BertTokenizerFast.from_pretrained(BASE_MODEL)
    model     = BertForSequenceClassification.from_pretrained(
                    BASE_MODEL, num_labels=3, output_attentions=True)

model.to(DEVICE)
model.eval()

# BERT has 12 layers, each with 12 attention heads = 144 attention patterns
print(f"\nBERT architecture:")
print(f"  Layers        : 12")
print(f"  Attention heads per layer: 12")
print(f"  Total attention patterns : 144")


# ============================================================
# STEP 2: The 10 sample reviews for analysis
# ============================================================
# The project requires minimum 10 samples with observations.
# Each entry: (sentence, aspect, expected_sentiment)

SAMPLES = [
    # Restaurant reviews
    ("The food was absolutely delicious and well-presented.",
     "food", "positive"),

    ("Service was painfully slow and the staff seemed uninterested.",
     "service", "negative"),

    ("The ambiance was lovely but the pasta was just average.",
     "pasta", "neutral"),

    ("I loved the dessert — richest chocolate cake I have ever had.",
     "dessert", "positive"),

    ("The wine selection is decent, nothing to write home about.",
     "wine selection", "neutral"),

    # Laptop / Tech reviews
    ("Battery life is outstanding — easily lasts a full day.",
     "battery life", "positive"),

    ("The keyboard feels mushy and completely unresponsive.",
     "keyboard", "negative"),

    ("Screen resolution is razor sharp and colors are vivid.",
     "screen resolution", "positive"),

    ("The fan noise is distractingly loud during heavy tasks.",
     "fan noise", "negative"),

    ("Build quality feels solid but the trackpad is just okay.",
     "trackpad", "neutral"),
]


# ============================================================
# STEP 3: Get attention weights from BERT
# ============================================================

def get_attention_and_prediction(text, aspect, model, tokenizer, device):
    """
    Runs one forward pass and returns:
      - tokens         : list of token strings
      - attentions     : tuple of tensors, one per layer
                         shape: (1, num_heads, seq_len, seq_len)
      - predicted label: string
      - confidence     : float
    """
    encoding = tokenizer(
        text, aspect,
        return_tensors='pt',
        max_length=128,
        padding=True,
        truncation=True,
    )

    input_ids      = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)
    token_type_ids = encoding.get('token_type_ids',
                     torch.zeros_like(input_ids)).to(device)

    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )

    tokens     = tokenizer.convert_ids_to_tokens(input_ids[0])
    attentions = outputs.attentions   # tuple of 12 tensors

    probs      = torch.softmax(outputs.logits, dim=-1).squeeze()
    pred_id    = probs.argmax().item()
    label      = ID2LABEL[pred_id]
    confidence = probs[pred_id].item()

    return tokens, attentions, label, confidence, probs


# ============================================================
# STEP 4: Static attention heatmap (for PDF report)
# ============================================================

def plot_attention_heatmap(tokens, attentions, layer=11, head=0,
                           title='', pred_label='', confidence=0.0,
                           save_path=None):
    """
    Plots a single attention head as a heatmap.

    Args:
      layer : which BERT layer to visualize (0-11, 11 = last)
      head  : which attention head (0-11)
    """
    # Extract attention matrix for chosen layer and head
    # Shape: (seq_len, seq_len)
    attn_matrix = attentions[layer][0, head].cpu().numpy()

    # Clean up token display (remove ## from WordPiece sub-tokens)
    clean_tokens = [t.replace('##', '') for t in tokens]

    fig, ax = plt.subplots(figsize=(max(8, len(tokens)*0.5),
                                    max(6, len(tokens)*0.4)))

    sns.heatmap(
        attn_matrix,
        xticklabels=clean_tokens,
        yticklabels=clean_tokens,
        cmap='Blues',
        ax=ax,
        linewidths=0.3,
        linecolor='lightgrey',
        vmin=0, vmax=attn_matrix.max(),
    )

    sentiment_color = {'positive': '🟢', 'neutral': '🟡', 'negative': '🔴'}.get(pred_label, '')
    ax.set_title(
        f"{title}\n"
        f"Prediction: {sentiment_color} {pred_label.upper()} ({confidence:.1%}) | "
        f"Layer {layer+1}, Head {head+1}",
        fontsize=10, pad=10
    )
    ax.set_xlabel('Key (what is attended to)', fontsize=9)
    ax.set_ylabel('Query (which token is attending)', fontsize=9)
    ax.tick_params(axis='x', rotation=45, labelsize=8)
    ax.tick_params(axis='y', rotation=0,  labelsize=8)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    plt.close()


# ============================================================
# STEP 5: Aspect-focused attention score
# ============================================================
# For each sample, we want to know: how much does the model
# attend to the aspect tokens when making its prediction?
# Higher aspect attention = model is "looking at the right place"

def compute_aspect_attention_score(tokens, attentions, aspect_tokens):
    """
    Computes the average attention weight directed at aspect tokens
    across ALL layers and ALL heads (last layer weighted more).

    Returns a score between 0 and 1.
    """
    # Find positions of aspect tokens in the token list
    aspect_positions = []
    aspect_lower = [a.lower().replace('##', '') for a in aspect_tokens]
    tokens_lower  = [t.lower().replace('##', '') for t in tokens]

    for i, tok in enumerate(tokens_lower):
        if tok in aspect_lower:
            aspect_positions.append(i)

    if not aspect_positions:
        return 0.0

    total_score = 0.0
    total_weight = 0.0

    for layer_idx, layer_attn in enumerate(attentions):
        # Weight later layers more (they are more task-relevant)
        layer_weight = (layer_idx + 1) / len(attentions)
        # Average across heads: shape (seq_len, seq_len)
        avg_head_attn = layer_attn[0].mean(dim=0).cpu().numpy()

        # Sum attention from all tokens TO aspect positions
        aspect_attn = avg_head_attn[:, aspect_positions].sum()
        total_score  += layer_weight * aspect_attn
        total_weight += layer_weight

    return float(total_score / (total_weight + 1e-8))


# ============================================================
# STEP 6: Run analysis on all 10 samples
# ============================================================

print("\n" + "="*60)
print("ATTENTION ANALYSIS — 10 SAMPLE REVIEWS")
print("="*60)

os.makedirs('attention_plots', exist_ok=True)

results = []
for idx, (text, aspect, expected) in enumerate(SAMPLES):
    tokens, attentions, pred_label, confidence, probs = \
        get_attention_and_prediction(text, aspect, model, tokenizer, DEVICE)

    # Tokenize aspect alone to find its tokens
    aspect_enc    = tokenizer.tokenize(aspect)
    aspect_score  = compute_aspect_attention_score(tokens, attentions, aspect_enc)

    correct = (pred_label == expected)
    emoji   = '✅' if correct else '❌'

    print(f"\nSample {idx+1:2d}: {text}")
    print(f"  Aspect   : '{aspect}'")
    print(f"  Expected : {expected:<10}  Predicted: {pred_label:<10} {emoji}  "
          f"Confidence: {confidence:.1%}")
    print(f"  Aspect attention score: {aspect_score:.4f}")

    results.append({
        'sample_id':     idx + 1,
        'text':          text,
        'aspect':        aspect,
        'expected':      expected,
        'predicted':     pred_label,
        'correct':       correct,
        'confidence':    confidence,
        'aspect_score':  aspect_score,
        'prob_positive': probs[0].item(),
        'prob_neutral':  probs[1].item(),
        'prob_negative': probs[2].item(),
    })

    # Save heatmap for each sample (last layer, head 0)
    save_path = f'attention_plots/sample_{idx+1:02d}_{aspect.replace(" ","_")}.png'
    plot_attention_heatmap(
        tokens, attentions,
        layer=11, head=0,
        title=f'Sample {idx+1}: "{text[:60]}..." | Aspect: {aspect}',
        pred_label=pred_label,
        confidence=confidence,
        save_path=save_path,
    )

results_df = pd.DataFrame(results) if 'pd' in dir() else None


# ============================================================
# STEP 7: Summary dashboard plot
# ============================================================

import pandas as pd
results_df = pd.DataFrame(results)

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Attention Interpretability — 10 Sample Summary', fontsize=14, fontweight='bold')

# Plot 1: Confidence per sample, colored by correct/incorrect
ax = axes[0, 0]
colors_bar = ['#2ecc71' if r['correct'] else '#e74c3c' for r in results]
bars = ax.bar(results_df['sample_id'], results_df['confidence'], color=colors_bar, edgecolor='black')
ax.set_title('Prediction Confidence per Sample\n(green=correct, red=incorrect)')
ax.set_xlabel('Sample')
ax.set_ylabel('Confidence')
ax.set_ylim(0, 1)
ax.axhline(0.5, color='gray', linestyle='--', alpha=0.5, label='50% threshold')
ax.set_xticks(results_df['sample_id'])
ax.legend()

# Plot 2: Aspect attention score per sample
ax = axes[0, 1]
ax.bar(results_df['sample_id'], results_df['aspect_score'],
       color='#5dade2', edgecolor='black')
ax.set_title('Aspect Attention Score per Sample\n(higher = model focuses on aspect)')
ax.set_xlabel('Sample')
ax.set_ylabel('Attention Score')
ax.set_xticks(results_df['sample_id'])

# Plot 3: Stacked bar of sentiment probabilities
ax = axes[1, 0]
width = 0.6
ax.bar(results_df['sample_id'], results_df['prob_positive'],
       width, label='Positive', color='#2ecc71', edgecolor='black')
ax.bar(results_df['sample_id'], results_df['prob_neutral'],
       width, bottom=results_df['prob_positive'],
       label='Neutral', color='#f39c12', edgecolor='black')
ax.bar(results_df['sample_id'], results_df['prob_negative'],
       width, bottom=results_df['prob_positive'] + results_df['prob_neutral'],
       label='Negative', color='#e74c3c', edgecolor='black')
ax.set_title('Sentiment Probability Distribution per Sample')
ax.set_xlabel('Sample')
ax.set_ylabel('Probability')
ax.set_ylim(0, 1)
ax.set_xticks(results_df['sample_id'])
ax.legend(loc='upper right', fontsize=8)

# Plot 4: Accuracy summary
ax = axes[1, 1]
correct_count   = results_df['correct'].sum()
incorrect_count = len(results_df) - correct_count
ax.pie([correct_count, incorrect_count],
       labels=[f'Correct ({correct_count})', f'Incorrect ({incorrect_count})'],
       colors=['#2ecc71', '#e74c3c'],
       autopct='%1.0f%%', startangle=90,
       textprops={'fontsize': 12})
ax.set_title(f'Overall Accuracy on 10 Samples\n({correct_count}/{len(results_df)} correct)')

plt.tight_layout()
plt.savefig('attention_summary.png', dpi=150, bbox_inches='tight')
plt.show()
print("\n✅ Summary dashboard saved to 'attention_summary.png'")


# ============================================================
# STEP 8: BertViz interactive visualization (if installed)
# ============================================================
# BertViz generates an interactive HTML page where you can
# click through all 144 attention heads in your browser.
# Great for your interpretability analysis section.

if BERTVIZ_AVAILABLE:
    print("\n🔍 Generating BertViz interactive visualizations...")

    # Pick one interesting sample (multi-aspect, conflicting sentiment)
    demo_text   = "The food was delicious but the service was dreadful."
    demo_aspect = "food"

    inputs = tokenizer(demo_text, demo_aspect, return_tensors='pt',
                       max_length=128, truncation=True)
    input_ids = inputs['input_ids'].to(DEVICE)

    with torch.no_grad():
        outputs = model(input_ids, output_attentions=True)

    attention = outputs.attentions   # tuple of 12 tensors
    tokens    = tokenizer.convert_ids_to_tokens(input_ids[0])

    # Head view: one head at a time — good for focused analysis
    head_view(attention, tokens)

    # Model view: all heads at once — good for overview
    model_view(attention, tokens)

    print("   BertViz visualizations rendered in your Jupyter notebook.")
    print("   If running as a script, use: jupyter notebook and call head_view()")
else:
    print("\n📌 BertViz not available.")
    print("   Install with: pip install bertviz")
    print("   Static heatmaps saved to attention_plots/ folder.")


# ============================================================
# STEP 9: Interpretability observations (template for report)
# ============================================================
# This prints a template for the observations you need to write
# in your 6-8 page report. Fill in based on YOUR actual results.

print("\n" + "="*60)
print("INTERPRETABILITY OBSERVATIONS TEMPLATE (for your report)")
print("="*60)
print("""
For each sample, observe and document:

1. TOKEN FOCUS
   - Which tokens receive the highest attention from [CLS]?
   - Does the model look at the aspect term directly?
   - Does it attend to sentiment-carrying words (great, terrible)?

2. ASPECT-SENTIMENT LINK
   - Is there a clear attention path from the aspect to its
     sentiment modifier? (e.g., "battery life" → "excellent")
   - In multi-aspect sentences, does the model correctly
     separate attention for each aspect?

3. CROSS-SENTENCE PATTERNS
   - Conjunctions like "but" — does attention shift after them?
   - Negation words like "not" — does the model attend to them?

4. HEAD SPECIALIZATION (from BertViz)
   - Some heads in early layers track syntax (subject-verb)
   - Some heads in later layers track semantics (word meaning)
   - Heads that point from [CLS] to sentiment words are most
     useful for classification

5. ERROR ANALYSIS
   - For wrong predictions: where did the model attend?
   - Was it distracted by other sentiment words?
   - Did sarcasm or irony confuse it?

Sample observation template:
  Sample 3 ("The ambiance was lovely but the pasta was average"):
  - Aspect: 'pasta' — aspect attention score: X.XX
  - Layer 12, Head 5 shows strong attention from 'pasta' to 'average'
  - The word 'but' creates an attention boundary between the two clauses
  - Model correctly ignores 'lovely' (relates to ambiance, not pasta)
  - Prediction: NEUTRAL ✅  Confidence: XX%
""")

print("🎉 Attention visualization complete!")
print("   Heatmaps saved to: attention_plots/")
print("   Summary plot saved to: attention_summary.png")
print("\n   Next step: Streamlit dashboard!")
