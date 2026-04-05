# ============================================================
# Aspect-Based Sentiment Analysis — Streamlit Dashboard
# ============================================================
# What this app does:
#   - User pastes any product/restaurant review
#   - App automatically extracts aspect terms using spaCy
#   - Runs each aspect through the fine-tuned BERT model
#   - Displays per-aspect sentiment scores + visualizations
#
# How to run:
#   pip install streamlit transformers torch spacy matplotlib
#               seaborn plotly
#   python -m spacy download en_core_web_sm
#   streamlit run streamlit_app.py
#
# To deploy on Hugging Face Spaces or Streamlit Community Cloud:
#   - Push this file + requirements.txt to GitHub
#   - Connect repo to Streamlit Cloud or HF Spaces
# ============================================================

import streamlit as st
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from transformers import BertTokenizerFast, BertForSequenceClassification
import spacy
import os
import time
from collections import defaultdict

# ============================================================
# PAGE CONFIG — must be first Streamlit command
# ============================================================

st.set_page_config(
    page_title="ABSA — Aspect Sentiment Analyzer",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# CUSTOM CSS — clean dark analytical theme
# ============================================================

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600&display=swap');

html, body, [class*="css"] {
    font-family: 'IBM Plex Sans', sans-serif;
}

.main { background-color: #0f1117; }

h1, h2, h3 {
    font-family: 'IBM Plex Mono', monospace !important;
    letter-spacing: -0.5px;
}

.aspect-card {
    background: #1a1d27;
    border-radius: 10px;
    padding: 18px 22px;
    margin: 10px 0;
    border-left: 4px solid #444;
    transition: transform 0.2s;
}
.aspect-card:hover { transform: translateX(4px); }
.aspect-card.positive { border-left-color: #00d68f; }
.aspect-card.negative { border-left-color: #ff4d6d; }
.aspect-card.neutral  { border-left-color: #f0b429; }

.aspect-term {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 1.1rem;
    font-weight: 600;
    color: #e8e8f0;
}
.sentiment-badge {
    display: inline-block;
    padding: 3px 12px;
    border-radius: 20px;
    font-size: 0.78rem;
    font-weight: 600;
    letter-spacing: 1px;
    text-transform: uppercase;
}
.badge-positive { background: #003d29; color: #00d68f; }
.badge-negative { background: #3d0011; color: #ff4d6d; }
.badge-neutral  { background: #3d2e00; color: #f0b429; }

.confidence-bar-bg {
    background: #2a2d3a;
    border-radius: 4px;
    height: 6px;
    margin-top: 8px;
    overflow: hidden;
}
.confidence-bar-fill {
    height: 100%;
    border-radius: 4px;
    transition: width 0.8s ease;
}

.metric-box {
    background: #1a1d27;
    border-radius: 8px;
    padding: 16px;
    text-align: center;
    border: 1px solid #2a2d3a;
}
.metric-number {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 2rem;
    font-weight: 600;
}
.metric-label {
    font-size: 0.8rem;
    color: #888;
    text-transform: uppercase;
    letter-spacing: 1px;
}

.stTextArea textarea {
    background: #1a1d27 !important;
    color: #e8e8f0 !important;
    border: 1px solid #2a2d3a !important;
    font-family: 'IBM Plex Sans', sans-serif !important;
    font-size: 0.95rem !important;
}

.highlight-aspect {
    background: rgba(0, 214, 143, 0.15);
    border-radius: 3px;
    padding: 1px 4px;
    font-weight: 600;
}
</style>
""", unsafe_allow_html=True)


# ============================================================
# CONSTANTS
# ============================================================

LABEL2ID = {'positive': 0, 'neutral': 1, 'negative': 2}
ID2LABEL  = {v: k for k, v in LABEL2ID.items()}
DEVICE    = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

SENTIMENT_COLORS = {
    'positive': '#00d68f',
    'neutral':  '#f0b429',
    'negative': '#ff4d6d',
}
SENTIMENT_EMOJI = {
    'positive': '😊',
    'neutral':  '😐',
    'negative': '😞',
}

MODEL_PATH = 'bert_absa_model/best_model'
BASE_MODEL  = 'bert-base-uncased'

# Demo reviews for quick testing
DEMO_REVIEWS = {
    "Restaurant review": (
        "The pasta was absolutely divine and portions were generous, "
        "but the service was shockingly slow and our waiter seemed disinterested. "
        "The ambiance was cozy and the wine selection was decent."
    ),
    "Laptop review": (
        "Battery life is outstanding — easily lasts 12 hours. "
        "The keyboard feels premium and responsive. "
        "However, the fan is annoyingly loud under load and the trackpad is just average."
    ),
    "Hotel review": (
        "The room was spotless and beautifully decorated. "
        "The bed was incredibly comfortable. "
        "Breakfast was disappointing — cold eggs and limited options. "
        "Staff at reception were friendly and helpful."
    ),
}


# ============================================================
# MODEL LOADING (cached so it only loads once)
# ============================================================

@st.cache_resource
def load_model():
    """Loads BERT model and tokenizer. Cached across sessions."""
    if os.path.exists(MODEL_PATH):
        tokenizer = BertTokenizerFast.from_pretrained(MODEL_PATH)
        model     = BertForSequenceClassification.from_pretrained(
                        MODEL_PATH, output_attentions=True)
        source = "fine-tuned"
    else:
        tokenizer = BertTokenizerFast.from_pretrained(BASE_MODEL)
        model     = BertForSequenceClassification.from_pretrained(
                        BASE_MODEL, num_labels=3, output_attentions=True)
        source = "base (not fine-tuned)"

    model.to(DEVICE)
    model.eval()
    return model, tokenizer, source


@st.cache_resource
def load_spacy():
    """Loads spaCy NLP model for aspect extraction."""
    try:
        return spacy.load('en_core_web_sm')
    except OSError:
        st.error("spaCy model not found. Run: python -m spacy download en_core_web_sm")
        return None


# ============================================================
# ASPECT EXTRACTION (rule-based with spaCy)
# ============================================================

def extract_aspects_spacy(text, nlp):
    """
    Rule-based aspect extraction using spaCy dependency parsing.

    Extracts:
      - Nouns and noun phrases (likely aspects)
      - Filters out pronouns and very common words

    Returns list of (aspect_term, sentence) pairs.
    """
    doc = nlp(text)
    aspects = []
    seen    = set()

    STOPWORDS = {'i', 'we', 'you', 'they', 'it', 'this', 'that',
                 'there', 'here', 'thing', 'everything', 'nothing'}

    for sent in doc.sents:
        for chunk in sent.noun_chunks:
            term = chunk.root.lemma_.lower()
            if (term not in STOPWORDS
                    and len(term) > 2
                    and term not in seen
                    and chunk.root.pos_ in ('NOUN', 'PROPN')):
                aspects.append({
                    'term':     chunk.text,
                    'sentence': sent.text.strip(),
                })
                seen.add(term)

    return aspects


def extract_aspects_manual(text):
    """
    Fallback: user manually enters aspect terms.
    Splits comma-separated input.
    """
    return []


# ============================================================
# SENTIMENT PREDICTION
# ============================================================

def predict_sentiment(text, aspect, model, tokenizer):
    """
    Returns sentiment prediction dict for (text, aspect) pair.
    """
    encoding = tokenizer(
        text, aspect,
        max_length=128,
        padding='max_length',
        truncation=True,
        return_tensors='pt',
    )

    with torch.no_grad():
        outputs = model(
            input_ids=encoding['input_ids'].to(DEVICE),
            attention_mask=encoding['attention_mask'].to(DEVICE),
            token_type_ids=encoding.get(
                'token_type_ids',
                torch.zeros(1, 128, dtype=torch.long)
            ).to(DEVICE),
        )

    probs      = torch.softmax(outputs.logits, dim=-1).squeeze()
    pred_id    = probs.argmax().item()
    label      = ID2LABEL[pred_id]
    confidence = probs[pred_id].item()
    attentions = outputs.attentions

    return {
        'sentiment':     label,
        'confidence':    confidence,
        'prob_positive': probs[0].item(),
        'prob_neutral':  probs[1].item(),
        'prob_negative': probs[2].item(),
        'attentions':    attentions,
        'tokens':        tokenizer.convert_ids_to_tokens(encoding['input_ids'][0]),
    }


# ============================================================
# VISUALIZATION HELPERS
# ============================================================

def make_radar_chart(results):
    """
    Radar chart showing positive/neutral/negative balance
    across all aspects.
    """
    aspects  = [r['term'] for r in results]
    pos_vals = [r['prob_positive'] for r in results]
    neu_vals = [r['prob_neutral']  for r in results]
    neg_vals = [r['prob_negative'] for r in results]

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(r=pos_vals, theta=aspects,
                  fill='toself', name='Positive',
                  line_color='#00d68f', fillcolor='rgba(0,214,143,0.15)'))
    fig.add_trace(go.Scatterpolar(r=neg_vals, theta=aspects,
                  fill='toself', name='Negative',
                  line_color='#ff4d6d', fillcolor='rgba(255,77,109,0.15)'))
    fig.add_trace(go.Scatterpolar(r=neu_vals, theta=aspects,
                  fill='toself', name='Neutral',
                  line_color='#f0b429', fillcolor='rgba(240,180,41,0.10)'))

    fig.update_layout(
        polar=dict(
            bgcolor='#1a1d27',
            radialaxis=dict(visible=True, range=[0, 1],
                            gridcolor='#2a2d3a', color='#888'),
            angularaxis=dict(gridcolor='#2a2d3a', color='#ccc'),
        ),
        paper_bgcolor='#0f1117',
        plot_bgcolor='#0f1117',
        font=dict(color='#ccc', family='IBM Plex Mono'),
        legend=dict(bgcolor='#1a1d27', bordercolor='#2a2d3a'),
        margin=dict(l=60, r=60, t=30, b=30),
    )
    return fig


def make_heatmap_plotly(results):
    """
    Heatmap of sentiment probabilities for all aspects.
    """
    aspects = [r['term'] for r in results]
    matrix  = np.array([
        [r['prob_positive'], r['prob_neutral'], r['prob_negative']]
        for r in results
    ])

    fig = go.Figure(data=go.Heatmap(
        z=matrix,
        x=['Positive', 'Neutral', 'Negative'],
        y=aspects,
        colorscale=[
            [0.0, '#1a1d27'],
            [0.5, '#5a3e8a'],
            [1.0, '#00d68f'],
        ],
        text=np.round(matrix, 2),
        texttemplate='%{text}',
        textfont=dict(size=12, family='IBM Plex Mono'),
        showscale=True,
    ))

    fig.update_layout(
        paper_bgcolor='#0f1117',
        plot_bgcolor='#1a1d27',
        font=dict(color='#ccc', family='IBM Plex Mono'),
        xaxis=dict(side='top'),
        margin=dict(l=20, r=20, t=40, b=20),
        height=max(200, len(aspects) * 50 + 80),
    )
    return fig


def make_attention_heatmap(tokens, attentions, layer=11, head=0):
    """
    Static matplotlib attention heatmap for one layer/head.
    Returns a matplotlib figure.
    """
    attn = attentions[layer][0, head].cpu().numpy()
    clean_tokens = [t.replace('##', '') for t in tokens if t != '[PAD]']
    n = len(clean_tokens)
    attn = attn[:n, :n]

    fig, ax = plt.subplots(figsize=(max(5, n*0.45), max(4, n*0.35)))
    fig.patch.set_facecolor('#0f1117')
    ax.set_facecolor('#1a1d27')

    sns.heatmap(attn, xticklabels=clean_tokens, yticklabels=clean_tokens,
                cmap='Blues', ax=ax, linewidths=0.3, linecolor='#2a2d3a',
                cbar_kws={'shrink': 0.6})

    ax.tick_params(axis='x', rotation=45, labelsize=7, colors='#aaa')
    ax.tick_params(axis='y', rotation=0,  labelsize=7, colors='#aaa')
    ax.set_title(f'Attention — Layer {layer+1}, Head {head+1}',
                 color='#ccc', fontsize=9, pad=8)
    plt.tight_layout()
    return fig


# ============================================================
# MAIN APP LAYOUT
# ============================================================

def main():
    # --- Header ---
    st.markdown("""
    <h1 style="color:#e8e8f0; margin-bottom:4px;">
        🔍 Aspect Sentiment Analyzer
    </h1>
    <p style="color:#666; font-size:0.9rem; margin-bottom:24px;">
        Fine-grained ABSA · BERT-base-uncased · SemEval 2014/2016
    </p>
    """, unsafe_allow_html=True)

    # --- Load models ---
    with st.spinner("Loading models..."):
        model, tokenizer, model_source = load_model()
        nlp = load_spacy()

    st.sidebar.markdown(f"""
    ### ⚙️ Model Info
    - **Model**: `bert-base-uncased`
    - **Source**: {model_source}
    - **Device**: `{DEVICE}`
    - **Classes**: positive · neutral · negative
    """)

    # --- Sidebar settings ---
    st.sidebar.markdown("### 🎛️ Settings")
    extraction_mode = st.sidebar.radio(
        "Aspect extraction",
        ["Auto (spaCy NLP)", "Manual input"],
        help="Auto uses dependency parsing. Manual lets you specify aspects."
    )
    show_attention = st.sidebar.checkbox("Show attention heatmap", value=False,
        help="Shows which tokens BERT focuses on. Slower to render.")
    attn_layer = st.sidebar.slider("Attention layer", 1, 12, 12) - 1
    attn_head  = st.sidebar.slider("Attention head",  1, 12, 1)  - 1

    st.sidebar.markdown("### 📋 Demo Reviews")
    demo_choice = st.sidebar.selectbox("Load a demo", ["— none —"] + list(DEMO_REVIEWS.keys()))

    # --- Input area ---
    col_input, col_aspects = st.columns([3, 1])

    with col_input:
        st.markdown("#### 📝 Enter a Review")
        default_text = DEMO_REVIEWS.get(demo_choice, "") if demo_choice != "— none —" else ""
        review_text  = st.text_area(
            label="review",
            value=default_text,
            height=140,
            placeholder="Paste any product or restaurant review here...",
            label_visibility="collapsed",
        )

    with col_aspects:
        st.markdown("#### ✏️ Aspects")
        if extraction_mode == "Manual input":
            manual_aspects = st.text_area(
                "One per line",
                height=140,
                placeholder="food\nservice\nambiance",
            )
        else:
            st.markdown(
                "<p style='color:#666;font-size:0.85rem;margin-top:12px;'>"
                "Aspects will be extracted automatically using spaCy "
                "dependency parsing.</p>",
                unsafe_allow_html=True,
            )
            manual_aspects = ""

    # --- Analyse button ---
    analyse_clicked = st.button("⚡ Analyse Sentiment", type="primary", use_container_width=True)

    if not analyse_clicked:
        st.markdown("""
        <div style="text-align:center;color:#444;padding:60px 0;font-family:'IBM Plex Mono',monospace;">
            ↑ Enter a review and click Analyse
        </div>
        """, unsafe_allow_html=True)
        return

    if not review_text.strip():
        st.warning("Please enter a review first.")
        return

    # --- Extract aspects ---
    if extraction_mode == "Manual input" and manual_aspects.strip():
        raw_aspects = [
            {'term': a.strip(), 'sentence': review_text}
            for a in manual_aspects.strip().split('\n')
            if a.strip()
        ]
    else:
        if nlp is None:
            st.error("spaCy not loaded. Switch to Manual input.")
            return
        raw_aspects = extract_aspects_spacy(review_text, nlp)

    if not raw_aspects:
        st.warning("No aspects found. Try Manual input mode.")
        return

    # --- Run predictions ---
    results = []
    progress = st.progress(0, text="Analysing aspects...")

    for i, asp in enumerate(raw_aspects):
        result = predict_sentiment(asp['sentence'], asp['term'], model, tokenizer)
        result['term']     = asp['term']
        result['sentence'] = asp['sentence']
        results.append(result)
        progress.progress((i + 1) / len(raw_aspects),
                          text=f"Analysing: '{asp['term']}'...")
        time.sleep(0.05)

    progress.empty()

    # ============================================================
    # RESULTS DISPLAY
    # ============================================================

    # --- Summary metrics ---
    pos_count = sum(1 for r in results if r['sentiment'] == 'positive')
    neg_count = sum(1 for r in results if r['sentiment'] == 'negative')
    neu_count = sum(1 for r in results if r['sentiment'] == 'neutral')
    avg_conf  = np.mean([r['confidence'] for r in results])

    overall = 'positive' if pos_count > neg_count else (
              'negative' if neg_count > pos_count else 'neutral')
    overall_color = SENTIMENT_COLORS[overall]

    st.markdown("---")
    st.markdown("#### 📊 Summary")
    m1, m2, m3, m4, m5 = st.columns(5)

    with m1:
        st.markdown(f"""
        <div class="metric-box">
            <div class="metric-number" style="color:{overall_color};">
                {SENTIMENT_EMOJI[overall]}
            </div>
            <div class="metric-label">Overall</div>
            <div style="color:{overall_color};font-size:0.85rem;font-weight:600;">
                {overall.upper()}
            </div>
        </div>""", unsafe_allow_html=True)

    for count, label, color in [
        (len(results), 'Aspects',  '#5dade2'),
        (pos_count,    'Positive', SENTIMENT_COLORS['positive']),
        (neg_count,    'Negative', SENTIMENT_COLORS['negative']),
        (neu_count,    'Neutral',  SENTIMENT_COLORS['neutral']),
    ]:
        with [m2, m3, m4, m5][['Aspects','Positive','Negative','Neutral'].index(label)]:
            st.markdown(f"""
            <div class="metric-box">
                <div class="metric-number" style="color:{color};">{count}</div>
                <div class="metric-label">{label}</div>
            </div>""", unsafe_allow_html=True)

    # --- Per-aspect cards + charts ---
    st.markdown("#### 🔎 Per-Aspect Results")
    col_cards, col_charts = st.columns([1, 1])

    with col_cards:
        for r in results:
            sent  = r['sentiment']
            color = SENTIMENT_COLORS[sent]
            conf  = r['confidence']
            bar_w = int(conf * 100)

            st.markdown(f"""
            <div class="aspect-card {sent}">
                <div style="display:flex;justify-content:space-between;align-items:center;">
                    <span class="aspect-term">"{r['term']}"</span>
                    <span class="sentiment-badge badge-{sent}">{sent}</span>
                </div>
                <div style="color:#888;font-size:0.8rem;margin-top:6px;">
                    {r['sentence'][:90]}{'...' if len(r['sentence'])>90 else ''}
                </div>
                <div style="display:flex;align-items:center;gap:10px;margin-top:8px;">
                    <div class="confidence-bar-bg" style="flex:1;">
                        <div class="confidence-bar-fill"
                             style="width:{bar_w}%;background:{color};"></div>
                    </div>
                    <span style="color:{color};font-size:0.8rem;font-family:'IBM Plex Mono',monospace;">
                        {conf:.0%}
                    </span>
                </div>
                <div style="display:flex;gap:12px;margin-top:6px;font-size:0.75rem;color:#666;">
                    <span>😊 {r['prob_positive']:.2f}</span>
                    <span>😐 {r['prob_neutral']:.2f}</span>
                    <span>😞 {r['prob_negative']:.2f}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

    with col_charts:
        if len(results) >= 3:
            st.markdown("**Sentiment Radar**")
            st.plotly_chart(make_radar_chart(results), use_container_width=True)
        else:
            st.markdown("**Probability Heatmap**")
            st.plotly_chart(make_heatmap_plotly(results), use_container_width=True)

    # Heatmap always shown below when >= 3 aspects
    if len(results) >= 3:
        st.markdown("**Probability Heatmap**")
        st.plotly_chart(make_heatmap_plotly(results), use_container_width=True)

    # --- Attention visualization ---
    if show_attention:
        st.markdown("---")
        st.markdown("#### 🧠 Attention Heatmap")

        aspect_choice = st.selectbox(
            "Select aspect to inspect",
            [r['term'] for r in results]
        )
        chosen = next(r for r in results if r['term'] == aspect_choice)

        if chosen['attentions']:
            fig = make_attention_heatmap(
                chosen['tokens'], chosen['attentions'],
                layer=attn_layer, head=attn_head
            )
            st.pyplot(fig, use_container_width=True)
            st.caption(
                f"Layer {attn_layer+1}, Head {attn_head+1} — "
                "rows = which token is looking, columns = what it attends to. "
                "Brighter = stronger attention."
            )

    # --- Download results ---
    st.markdown("---")
    import pandas as pd, io
    export_df = pd.DataFrame([{
        'aspect':      r['term'],
        'sentiment':   r['sentiment'],
        'confidence':  f"{r['confidence']:.2%}",
        'prob_pos':    f"{r['prob_positive']:.3f}",
        'prob_neu':    f"{r['prob_neutral']:.3f}",
        'prob_neg':    f"{r['prob_negative']:.3f}",
        'sentence':    r['sentence'],
    } for r in results])

    csv_buffer = io.StringIO()
    export_df.to_csv(csv_buffer, index=False)

    st.download_button(
        label="⬇️ Download Results as CSV",
        data=csv_buffer.getvalue(),
        file_name="absa_results.csv",
        mime="text/csv",
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == '__main__':
    main()
