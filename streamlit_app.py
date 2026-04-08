# ============================================================
# Aspect-Based Sentiment Analysis — Streamlit Dashboard (Fixed)
# ============================================================

import streamlit as st
import numpy as np
import os
import time
import re
import torch
from transformers import BertTokenizerFast, BertForSequenceClassification
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import pandas as pd
import io

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="ABSA — Aspect Sentiment Analyzer",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# CUSTOM CSS
# ============================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600&display=swap');
html, body, [class*="css"] { font-family: 'IBM Plex Sans', sans-serif; }
.main { background-color: #0f1117; }
h1, h2, h3 { font-family: 'IBM Plex Mono', monospace !important; letter-spacing: -0.5px; }
.aspect-card {
    background: #1a1d27; border-radius: 10px; padding: 18px 22px; margin: 10px 0;
    border-left: 4px solid #444; transition: transform 0.2s;
}
.aspect-card:hover { transform: translateX(4px); }
.aspect-card.positive { border-left-color: #00d68f; }
.aspect-card.negative { border-left-color: #ff4d6d; }
.aspect-card.neutral { border-left-color: #f0b429; }
.aspect-term { font-family: 'IBM Plex Mono', monospace; font-size: 1.1rem; font-weight: 600; color: #e8e8f0; }
.sentiment-badge {
    display: inline-block; padding: 3px 12px; border-radius: 20px;
    font-size: 0.78rem; font-weight: 600; letter-spacing: 1px; text-transform: uppercase;
}
.badge-positive { background: #003d29; color: #00d68f; }
.badge-negative { background: #3d0011; color: #ff4d6d; }
.badge-neutral { background: #3d2e00; color: #f0b429; }
.confidence-bar-bg { background: #2a2d3a; border-radius: 4px; height: 6px; margin-top: 8px; overflow: hidden; }
.confidence-bar-fill { height: 100%; border-radius: 4px; transition: width 0.8s ease; }
.metric-box {
    background: #1a1d27; border-radius: 8px; padding: 16px; text-align: center; border: 1px solid #2a2d3a;
}
.metric-number { font-family: 'IBM Plex Mono', monospace; font-size: 2rem; font-weight: 600; }
.metric-label { font-size: 0.8rem; color: #888; text-transform: uppercase; letter-spacing: 1px; }
</style>
""", unsafe_allow_html=True)

# ============================================================
# CONSTANTS
# ============================================================
LABEL2ID = {'positive': 0, 'neutral': 1, 'negative': 2}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}
SENTIMENT_COLORS = {'positive': '#00d68f', 'neutral': '#f0b429', 'negative': '#ff4d6d'}
SENTIMENT_EMOJI = {'positive': '😊', 'neutral': '😐', 'negative': '😞'}

MODEL_PATH = 'bert_absa_model/best_model'
BASE_MODEL = 'bert-base-uncased'

DEMO_REVIEWS = {
    "Restaurant review": "The pasta was absolutely divine and portions were generous, but the service was shockingly slow and our waiter seemed disinterested. The ambiance was cozy and the wine selection was decent.",
    "Laptop review": "Battery life is outstanding — easily lasts 12 hours. The keyboard feels premium and responsive. However, the fan is annoyingly loud under load and the trackpad is just average.",
    "Hotel review": "The room was spotless and beautifully decorated. The bed was incredibly comfortable. Breakfast was disappointing — cold eggs and limited options. Staff at reception were friendly and helpful."
}

# ============================================================
# MODEL LOADING
# ============================================================
@st.cache_resource
def load_model():
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if os.path.exists(MODEL_PATH):
        tokenizer = BertTokenizerFast.from_pretrained(MODEL_PATH)
        model = BertForSequenceClassification.from_pretrained(MODEL_PATH, output_attentions=True)
        source = "fine-tuned"
    else:
        tokenizer = BertTokenizerFast.from_pretrained(BASE_MODEL)
        model = BertForSequenceClassification.from_pretrained(BASE_MODEL, num_labels=3, output_attentions=True)
        source = "base (not fine-tuned)"
    
    model.to(DEVICE)
    model.eval()
    return model, tokenizer, source, DEVICE

# ============================================================
# LIGHTWEIGHT ASPECT EXTRACTION
# ============================================================
def extract_aspects_light(text):
    sentences = re.split(r'[.!?]+', text)
    aspects = []
    seen = set()
    common_aspects = {'food', 'service', 'staff', 'place', 'room', 'battery', 'keyboard', 'screen',
                      'camera', 'price', 'quality', 'ambiance', 'taste', 'portion'}
    
    for sent in sentences:
        sent = sent.strip()
        if not sent: continue
        words = re.findall(r'\b\w+\b', sent.lower())
        for word in words:
            if len(word) > 3 and word not in seen:
                if word in common_aspects or any(noun in word for noun in ['ing', 'ment', 'tion', 'ness']):
                    aspects.append({'term': word.capitalize(), 'sentence': sent})
                    seen.add(word)
    if not aspects:
        aspects.append({'term': 'Overall Experience', 'sentence': text})
    return aspects[:8]

# ============================================================
# SENTIMENT PREDICTION
# ============================================================
def predict_sentiment(text, aspect, model, tokenizer, DEVICE):
    encoding = tokenizer(
        text, aspect,
        max_length=128,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )
    
    with torch.no_grad():
        outputs = model(
            input_ids=encoding['input_ids'].to(DEVICE),
            attention_mask=encoding['attention_mask'].to(DEVICE),
            token_type_ids=encoding.get('token_type_ids', torch.zeros(1, 128, dtype=torch.long)).to(DEVICE)
        )
    
    probs = torch.softmax(outputs.logits, dim=-1).squeeze()
    pred_id = probs.argmax().item()
    label = ID2LABEL[pred_id]
    confidence = probs[pred_id].item()
    
    return {
        'sentiment': label,
        'confidence': confidence,
        'prob_positive': probs[0].item(),
        'prob_neutral': probs[1].item(),
        'prob_negative': probs[2].item(),
        'attentions': outputs.attentions,
        'tokens': tokenizer.convert_ids_to_tokens(encoding['input_ids'][0]),
        'term': aspect,
        'sentence': text
    }

# ============================================================
# VISUALIZATIONS
# ============================================================
def make_radar_chart(results):
    aspects = [r['term'] for r in results]
    pos_vals = [r['prob_positive'] for r in results]
    neu_vals = [r['prob_neutral'] for r in results]
    neg_vals = [r['prob_negative'] for r in results]
    
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(r=pos_vals, theta=aspects, fill='toself', name='Positive',
                                  line_color='#00d68f', fillcolor='rgba(0,214,143,0.15)'))
    fig.add_trace(go.Scatterpolar(r=neg_vals, theta=aspects, fill='toself', name='Negative',
                                  line_color='#ff4d6d', fillcolor='rgba(255,77,109,0.15)'))
    fig.add_trace(go.Scatterpolar(r=neu_vals, theta=aspects, fill='toself', name='Neutral',
                                  line_color='#f0b429', fillcolor='rgba(240,180,41,0.10)'))
    
    fig.update_layout(
        polar=dict(bgcolor='#1a1d27', radialaxis=dict(range=[0, 1])),
        paper_bgcolor='#0f1117', plot_bgcolor='#0f1117',
        font=dict(color='#ccc'), legend=dict(bgcolor='#1a1d27')
    )
    return fig

def make_heatmap_plotly(results):
    aspects = [r['term'] for r in results]
    matrix = np.array([[r['prob_positive'], r['prob_neutral'], r['prob_negative']] for r in results])
    
    fig = go.Figure(data=go.Heatmap(
        z=matrix, x=['Positive', 'Neutral', 'Negative'], y=aspects,
        colorscale=[[0.0, '#1a1d27'], [0.5, '#5a3e8a'], [1.0, '#00d68f']],
        text=np.round(matrix, 2), texttemplate='%{text}'
    ))
    fig.update_layout(paper_bgcolor='#0f1117', plot_bgcolor='#1a1d27', font=dict(color='#ccc'))
    return fig

# ============================================================
# MAIN APP
# ============================================================
def main():
    st.markdown("""
    <h1 style="color:#e8e8f0; margin-bottom:4px;">🔍 Aspect Sentiment Analyzer</h1>
    <p style="color:#666; font-size:0.9rem;">Fine-grained ABSA using BERT • Lightweight aspect extraction</p>
    """, unsafe_allow_html=True)

    # Load model
    with st.spinner("Loading BERT model..."):
        model, tokenizer, model_source, DEVICE = load_model()

    st.sidebar.markdown(f"**Model**: bert-base-uncased  \n**Source**: {model_source}  \n**Device**: {DEVICE}")

    extraction_mode = st.sidebar.radio("Aspect extraction", 
                                       ["Auto (Lightweight)", "Manual input"], index=0)
    show_attention = st.sidebar.checkbox("Show attention heatmap", value=False)

    demo_choice = st.sidebar.selectbox("Load demo review", ["— none —"] + list(DEMO_REVIEWS.keys()))
    default_text = DEMO_REVIEWS.get(demo_choice, "") if demo_choice != "— none —" else ""

    review_text = st.text_area("📝 Enter a Review", value=default_text, height=160,
                               placeholder="Paste your product or restaurant review here...")

    if st.button("⚡ Analyse Sentiment", type="primary", use_container_width=True):
        if not review_text.strip():
            st.warning("Please enter a review.")
            return

        # Extract aspects
        if extraction_mode == "Manual input":
            manual_input = st.text_input("Enter aspects (comma separated)", "food, service, ambiance")
            raw_aspects = [{'term': a.strip(), 'sentence': review_text} 
                          for a in manual_input.split(',') if a.strip()]
        else:
            raw_aspects = extract_aspects_light(review_text)

        if not raw_aspects:
            st.warning("No aspects found. Try Manual mode.")
            return

        # Run predictions
        results = []
        progress = st.progress(0, text="Analyzing aspects...")
        for i, asp in enumerate(raw_aspects):
            result = predict_sentiment(asp['sentence'], asp['term'], model, tokenizer, DEVICE)
            results.append(result)
            progress.progress((i + 1) / len(raw_aspects))
        progress.empty()

        st.success(f"Analysis complete! Found {len(results)} aspects.")

        # ====================== RESULTS DISPLAY ======================
        # Summary metrics
        pos_count = sum(1 for r in results if r['sentiment'] == 'positive')
        neg_count = sum(1 for r in results if r['sentiment'] == 'negative')
        neu_count = sum(1 for r in results if r['sentiment'] == 'neutral')
        overall = 'positive' if pos_count > neg_count else ('negative' if neg_count > pos_count else 'neutral')
        overall_color = SENTIMENT_COLORS[overall]

        st.markdown("---")
        st.markdown("#### 📊 Summary")
        m1, m2, m3, m4, m5 = st.columns(5)
        with m1:
            st.markdown(f"""
            <div class="metric-box">
                <div class="metric-number" style="color:{overall_color};">{SENTIMENT_EMOJI[overall]}</div>
                <div class="metric-label">Overall</div>
                <div style="color:{overall_color};font-weight:600;">{overall.upper()}</div>
            </div>""", unsafe_allow_html=True)

        for col, count, label, color in zip([m2, m3, m4, m5],
                                            [len(results), pos_count, neg_count, neu_count],
                                            ['Aspects', 'Positive', 'Negative', 'Neutral'],
                                            ['#5dade2', SENTIMENT_COLORS['positive'], 
                                             SENTIMENT_COLORS['negative'], SENTIMENT_COLORS['neutral']]):
            with col:
                st.markdown(f"""
                <div class="metric-box">
                    <div class="metric-number" style="color:{color};">{count}</div>
                    <div class="metric-label">{label}</div>
                </div>""", unsafe_allow_html=True)

        # Per-aspect cards
        st.markdown("#### 🔎 Per-Aspect Results")
        col_cards, col_charts = st.columns([1, 1])
        
        with col_cards:
            for r in results:
                sent = r['sentiment']
                color = SENTIMENT_COLORS[sent]
                conf = r['confidence']
                bar_w = int(conf * 100)
                st.markdown(f"""
                <div class="aspect-card {sent}">
                    <div style="display:flex;justify-content:space-between;align-items:center;">
                        <span class="aspect-term">"{r['term']}"</span>
                        <span class="sentiment-badge badge-{sent}">{sent}</span>
                    </div>
                    <div style="color:#888;font-size:0.85rem;margin-top:6px;">
                        {r['sentence'][:100]}{'...' if len(r['sentence']) > 100 else ''}
                    </div>
                    <div style="display:flex;align-items:center;gap:10px;margin-top:8px;">
                        <div class="confidence-bar-bg" style="flex:1;">
                            <div class="confidence-bar-fill" style="width:{bar_w}%;background:{color};"></div>
                        </div>
                        <span style="color:{color};font-family:'IBM Plex Mono',monospace;">{conf:.0%}</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)

        with col_charts:
            if len(results) >= 2:
                st.plotly_chart(make_radar_chart(results), use_container_width=True)
            st.plotly_chart(make_heatmap_plotly(results), use_container_width=True)

        # Attention heatmap (optional)
        if show_attention and results:
            st.markdown("---")
            st.markdown("#### 🧠 Attention Heatmap")
            aspect_choice = st.selectbox("Select aspect", [r['term'] for r in results])
            chosen = next(r for r in results if r['term'] == aspect_choice)
            if chosen['attentions']:
                # Simple attention display (you can enhance this)
                st.info("Attention visualization can be added here if needed.")

        # Download button
        export_df = pd.DataFrame([{
            'aspect': r['term'],
            'sentiment': r['sentiment'],
            'confidence': f"{r['confidence']:.2%}",
            'prob_positive': r['prob_positive'],
            'prob_neutral': r['prob_neutral'],
            'prob_negative': r['prob_negative'],
            'sentence': r['sentence']
        } for r in results])
        
        csv_buffer = io.StringIO()
        export_df.to_csv(csv_buffer, index=False)
        st.download_button(
            label="⬇️ Download Results as CSV",
            data=csv_buffer.getvalue(),
            file_name="absa_results.csv",
            mime="text/csv"
        )

if __name__ == '__main__':
    main()
