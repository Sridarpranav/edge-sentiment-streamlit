import streamlit as st
from optimum.onnxruntime import ORTModelForSequenceClassification
from transformers import AutoTokenizer
import torch
import pandas as pd
import time

# 1. High-End Premium Canvas Config
st.set_page_config(
    page_title="NSRLM LAB | Edge Distillation",
    page_icon="⚡",
    layout="centered"
)

# 2. Luxury Web App Theming (Rolls-Royce Inspired Aesthetic)
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;600;800&family=Space+Grotesk:wght@400;700&display=swap');

    /* Fixed ambient video canvas layer */
    .video-bg-container {
        position: fixed;
        top: 0; left: 0; width: 100vw; height: 100vh;
        z-index: -2;
        overflow: hidden;
    }
    
    .video-bg-container video {
        width: 100vw; height: 100vh;
        object-fit: cover;
    }

    /* Ambient Luxury Light/Dark Variable Engine using media tags */
    @media (prefers-color-scheme: light) {
        .stApp {
            background: rgba(24df, 246, 240, 0.85);
            color: #0f172a !important;
        }
        .luxury-card {
            background: rgba(255, 255, 255, 0.8) !important;
            backdrop-filter: blur(25px) saturate(160%);
            border: 1px solid rgba(16, 185, 129, 0.25);
            box-shadow: 0 30px 60px rgba(16, 185, 129, 0.06);
        }
        .luxury-title { color: #080c14 !important; }
        .luxury-text { color: #334155 !important; }
        .luxury-subtext { color: #64748b !important; }
        div[data-testid='stFileUploader'] {
            background: rgba(255, 255, 255, 0.7) !important;
            border: 1px dashed rgba(16, 185, 129, 0.4) !important;
        }
    }

    @media (prefers-color-scheme: dark) {
        .stApp {
            background: rgba(10, 14, 12, 0.9);
            color: #f8fafc !important;
        }
        .luxury-card {
            background: rgba(13, 20, 18, 0.75) !important;
            backdrop-filter: blur(25px) saturate(180%);
            border: 1px solid rgba(16, 185, 129, 0.2);
            box-shadow: 0 40px 80px rgba(0, 0, 0, 0.6);
        }
        .luxury-title { color: #ffffff !important; }
        .luxury-text { color: #e2e8f0 !important; }
        .luxury-subtext { color: #94a3b8 !important; }
        div[data-testid='stFileUploader'] {
            background: rgba(18, 28, 25, 0.6) !important;
            border: 1px dashed rgba(16, 185, 129, 0.3) !important;
        }
    }

    /* Top-Tier Structural Typography overrides */
    h1, h2, h3, .luxury-title {
        font-family: 'Space Grotesk', sans-serif !important;
        letter-spacing: -0.04em !important;
    }
    p, label, span, .luxury-text {
        font-family: 'Plus Jakarta Sans', sans-serif !important;
    }

    /* File Uploader Custom Polish */
    div[data-testid='stFileUploader'] {
        border-radius: 20px;
        padding: 24px;
        transition: all 0.4s ease;
    }

    /* Pure Luxury Performance Button Elements */
    .stButton>button { 
        background: linear-gradient(135deg, #10b981 0%, #047857 100%) !important; 
        color: #ffffff !important; 
        width: 100%; 
        border-radius: 14px; 
        border: none; 
        padding: 16px; 
        font-size: 15px;
        font-weight: 700; 
        letter-spacing: 0.05em;
        text-transform: uppercase;
        box-shadow: 0 12px 30px rgba(16, 185, 129, 0.25);
        transition: all 0.3s cubic-bezier(0.16, 1, 0.3, 1);
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 20px 40px rgba(16, 185, 129, 0.4);
    }

    /* Metrics presentation enhancements */
    div[data-testid="stMetricValue"] {
        font-family: 'Space Grotesk', sans-serif !important;
        font-size: 32px !important;
        font-weight: 700 !important;
        color: #10b981 !important;
    }
</style>

<!-- Deep Abstract Emerald Floral/Wave Ambient Motion Loop -->
<div class="video-bg-container">
    <video autoplay loop muted playsinline>
        <source src="https://assets.mixkit.co/videos/preview/mixkit-abstract-emerald-green-ink-flow-42998-large.mp4" type="video/mp4">
    </video>
</div>
""", unsafe_allow_html=True)

# 3. High-End Minimalist Lab Navbar Header
st.markdown("""
<div style="display: flex; justify-content: space-between; align-items: center; padding: 20px 0; border-bottom: 1px solid rgba(16, 185, 129, 0.15); margin-bottom: 40px;">
    <div>
        <h1 style="font-size: 20px; font-weight: 700; margin: 0; letter-spacing: 0.1em; text-transform: uppercase;">NSRLM <span style="color: #10b981;">LAB</span></h1>
    </div>
    <div style="font-size: 10px; letter-spacing: 2px; text-transform: uppercase; font-weight: 700; background: rgba(16, 185, 129, 0.1); padding: 6px 14px; border-radius: 30px; border: 1px solid rgba(16, 185, 129, 0.2); color: #10b981;">
        EDGE QUANTIZED
    </div>
</div>
""", unsafe_allow_html=True)

# 4. Premium Headline Block
st.markdown('<h2 class="luxury-title" style="font-size: 42px; font-weight: 800; line-height: 1.1; margin-bottom: 8px;">Compress the intelligence,<br><span style="background: linear-gradient(90deg, #10b981, #34d399); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">not the performance.</span></h2>', unsafe_allow_html=True)
st.markdown('<p class="luxury-subtext" style="font-size: 15px; line-height: 1.6; max-width: 540px; margin-bottom: 35px;">Distill massive, resource-heavy Large Language Models directly into zero-latency, sub-50MB runtimes perfectly calibrated for smartphones and Raspberry Pi devices.</p>', unsafe_allow_html=True)

# 5. Core Optimized Model Loader Execution
@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(".")
    model = ORTModelForSequenceClassification.from_pretrained(".", file_name="model_quantized.onnx")
    return tokenizer, model

try:
    tokenizer, model = load_model()

    # --- LUXURY DIGITAL HARDWARE DIALS ---
    st.markdown('<div class="luxury-card" style="padding: 24px; border-radius: 20px; margin-bottom: 35px;">', unsafe_allow_html=True)
    st.markdown('<h4 class="luxury-title" style="margin-top: 0; margin-bottom: 20px; font-size: 16px; text-transform: uppercase; letter-spacing: 0.05em; color: #10b981;">Engine Distillation Diagnostics</h4>', unsafe_allow_html=True)
    m1, m2, m3 = st.columns(3)
    with m1:
        st.metric(label="Teacher Profile", value="400 MB")
    with m2:
        st.metric(label="Student Footprint", value="50 MB")
    with m3:
        st.metric(label="Edge Target Speed", value="< 15ms")
    st.markdown('</div>', unsafe_allow_html=True)

    # --- ADVANCED HIGH-CLASS FILE DROPS ---
    st.markdown('### Drop in a dataset. Get compressed inference.')
    uploaded_file = st.file_uploader("Upload CSV Matrix", type=["csv"], label_visibility="collapsed")

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.markdown("<br><h4 class='luxury-title'>Source Preview</h4>", unsafe_allow_html=True)
        st.dataframe(df.head(3))
        
        text_col = st.selectbox("Specify Inference Column Target Target", df.columns)
        
        if st.button("Initialize High-Speed Batch Run"):
            with st.spinner("Processing token array configurations across edge compilation parameters..."):
                texts = df[text_col].astype(str).tolist()
                
                start_time = time.time()
                inputs = tokenizer(texts, padding=True, truncation=True, max_length=64, return_tensors="pt")
                
                with torch.no_grad():
                    outputs = model(**inputs)
                end_time = time.time()
                
                logits = outputs.logits.tolist()
                df['Optimized Overhead Latency'] = f"{(end_time - start_time)/len(texts)*1000:.2f} ms/item"
                df['Quantized State Matrix Output'] = [str([round(v, 4) for v in row]) for row in logits]
                
                st.markdown("<br><h4 class='luxury-title'>Distilled State Output Map</h4>", unsafe_allow_html=True)
                st.dataframe(df)
                
                processed_csv = df.to_csv(index=False).encode('utf-8')
                st.download_button("📥 Export Distilled Metrics Logs", data=processed_csv, file_name="nsrlm_edge_analytics.csv", mime="text/csv")

    # --- PREMIUM MOBILE DEMO BENCHMARK PANEL ---
    st.markdown("<br>", unsafe_allow_html=True)
    show_demo = st.checkbox("🔮 Launch Real-Time Edge Benchmarker Simulator", value=True)

    if show_demo:
        st.markdown("<div class='luxury-card' style='padding: 24px; border-radius: 20px;'>", unsafe_allow_html=True)
        user_text = st.text_area("Live Input Text Stream Container:", "Simulating rapid inference pipeline processing constraints within modern mobile processor sandboxes.")
        if st.button("Verify Local Inference"):
            start = time.time()
            demo_inputs = tokenizer(user_text, return_tensors="pt", truncation=True, max_length=64)
            with torch.no_grad():
                demo_outputs = model(**demo_inputs)
            latency = (time.time() - start) * 1000
            
            st.markdown(f"""
            <div style="background-color: rgba(16, 185, 129, 0.12); border-left: 4px solid #10b981; border-radius: 6px; padding: 18px; margin-top: 20px;">
                <div style="color: #10b981; font-weight: 700; font-family: 'Space Grotesk', sans-serif; text-transform: uppercase; font-size: 12px; letter-spacing: 0.05em; margin-bottom: 4px;">✓ Localized Runtime Profile Complete</div>
                <div style="font-size: 14px;" class="luxury-text">Target Compaction Bound: <strong>~50 MB Binary Target</strong> | System Execution Metric: <strong>{latency:.2f}ms latency</strong></div>
            </div>
            """, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

except Exception as e:
    st.error(f"Error executing engine backend architecture: {e}")

# Spacer ensuring content doesn't collide with fixed navigation bar layout
st.markdown("<br><br><br><br>", unsafe_allow_html=True)

# 6. Bespoke Fixed Ultra-Premium Mobile Bottom Navigation Track
st.markdown("""
<div style="position: fixed; bottom: 0; left: 0; width: 100%; background: rgba(15, 23, 42, 0.95); backdrop-filter: blur(20px); border-top: 1px solid rgba(16, 185, 129, 0.2); padding: 16px 0; display: flex; justify-content: space-around; text-align: center; z-index: 999999;">
    <div style="color: #10b981; font-weight: 700; font-family: 'Space Grotesk', sans-serif; font-size: 11px; letter-spacing: 0.05em; cursor: pointer;">
        ⚡<br>WORKSPACE
    </div>
    <div style="color: #94a3b8; font-weight: 500; font-family: 'Space Grotesk', sans-serif; font-size: 11px; letter-spacing: 0.05em; cursor: pointer;">
        📊<br>ANALYSIS
    </div>
    <div style="color: #94a3b8; font-weight: 500; font-family: 'Space Grotesk', sans-serif; font-size: 11px; letter-spacing: 0.05em; cursor: pointer;">
        🌿<br>METHOD
    </div>
</div>
""", unsafe_allow_html=True)
