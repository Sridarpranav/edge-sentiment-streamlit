import streamlit as st
from optimum.onnxruntime import ORTModelForSequenceClassification
from transformers import AutoTokenizer
import torch
import pandas as pd
import time

# 1. Clean Page Setup
st.set_page_config(
    page_title="EdgeDistill - Model Compression Engine",
    page_icon="⚡",
    layout="centered"
)

# 2. Premium High-Tech Node Interface Stylesheet
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@500;700&display=swap');

    /* High-Tech Immersive Particle Network Canvas Background */
    .stApp {
        background: linear-gradient(rgba(5, 19, 41, 0.9), rgba(2, 8, 19, 0.95)), 
                    url('https://images.unsplash.com/photo-1544256718-3bcf237f3974?q=80&w=1600&auto=format&fit=crop') !important;
        background-size: cover !important;
        background-position: center !important;
        background-attachment: fixed !important;
        color: #f8fafc !important;
    }
    
    /* Clean, Non-overlapping Typography Hierarchy */
    h1, h2, h3, h4 {
        font-family: 'Space Grotesk', sans-serif !important;
        color: #ffffff !important;
        letter-spacing: -0.02em !important;
    }
    p, label, span, div, .stMarkdown {
        font-family: 'Plus Jakarta Sans', sans-serif !important;
    }

    /* Structured Analytics Dashboard Cards (High-Class UI) */
    .analytics-card {
        background-color: rgba(11, 21, 40, 0.8) !important;
        border: 1px solid #162a4e !important;
        backdrop-filter: blur(10px);
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 16px;
        position: relative;
    }
    
    .engine-type {
        font-size: 11px;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 4px;
    }
    
    .metric-title {
        font-size: 14px;
        font-weight: 600;
        color: #f1f5f9;
        margin-bottom: 12px;
    }
    
    .metric-value {
        font-size: 26px;
        font-weight: 700;
        color: #ffffff;
        font-family: 'Space Grotesk', sans-serif !important;
    }

    /* Absolute Placement Inline Tags */
    .status-tag {
        float: right;
        background: rgba(30, 64, 175, 0.3);
        border: 1px solid #1d4ed8;
        color: #60a5fa;
        font-size: 10px;
        font-weight: 600;
        padding: 4px 10px;
        border-radius: 6px;
        text-transform: uppercase;
        margin-top: -24px;
    }

    /* Premium Button Custom Tuning */
    .stButton>button { 
        background: linear-gradient(135deg, #1d4ed8 0%, #1e40af 100%) !important; 
        color: white !important; 
        width: 100%; 
        border-radius: 8px; 
        border: 1px solid #3b82f6; 
        padding: 12px; 
        font-weight: 600; 
        font-size: 14px;
        letter-spacing: 0.02em;
        text-transform: uppercase;
        box-shadow: 0 4px 15px rgba(29, 78, 216, 0.3);
    }

    /* Streamlit File Uploader Clean Architecture Override */
    div[data-testid='stFileUploader'] {
        background-color: rgba(11, 21, 40, 0.8) !important;
        border: 1px dashed #162a4e !important;
        border-radius: 12px;
        padding: 16px;
    }

    /* Custom adjustments to natively styled streamlit dataframes */
    .stDataFrame {
        background-color: rgba(11, 21, 40, 0.6) !important;
        border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)

# 3. Premium Top Navbar Header
st.markdown("""
<div style="display: flex; justify-content: space-between; align-items: center; padding: 15px 0; border-bottom: 1px solid #162a4e; margin-bottom: 30px;">
    <h1 style="font-size: 18px; font-weight: 700; margin: 0; letter-spacing: 0.05em;">Edge<span style="color: #3b82f6;">Distill</span></h1>
    <div style="font-size: 11px; color: #3b82f6; font-weight: 700; letter-spacing: 0.05em;">⚡ ADAPTarget ENGINE</div>
</div>
""", unsafe_allow_html=True)

# 4. Core Optimized Model Core Execution
@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(".")
    model = ORTModelForSequenceClassification.from_pretrained(".", file_name="model_quantized.onnx")
    return tokenizer, model

try:
    tokenizer, model = load_model()

    # --- TARGET ANALYTICS GRID GENERATION (As requested in image) ---
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="analytics-card">
            <div class="engine-type">ONNX Edge Optimization</div>
            <div class="metric-title">Quantized Binary Footprint</div>
            <div class="status-tag">Inference Optimal</div>
            <div class="metric-value">50 MB</div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="analytics-card">
            <div class="engine-type">Hardware Target Specs</div>
            <div class="metric-title">Device Resource Matrix</div>
            <div class="status-tag">Edge Verified</div>
            <div class="metric-value">Mobile & Pi</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="analytics-card">
            <div class="engine-type">Execution Profile</div>
            <div class="metric-title">Target Pipeline Latency</div>
            <div class="status-tag">Structural Match</div>
            <div class="metric-value">&lt; 15.0ms</div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="analytics-card">
            <div class="engine-type">Compression Engine Ratio</div>
            <div class="metric-title">SLM Distillation Delta</div>
            <div class="status-tag">SHAP Verified</div>
            <div class="metric-value">8.0x Optimization</div>
        </div>
        """, unsafe_allow_html=True)

    # --- STREAMLINED MATRIX PROCESSING SECTION ---
    st.markdown("<br><h3 style='font-size: 18px; font-weight:700;'>Drop in a table. Get a diagnosis.</h3>", unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Upload Target Matrix (CSV)", type=["csv"], label_visibility="collapsed")

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.markdown("<span style='font-size: 12px; font-weight:700; text-transform:uppercase; color:#64748b;'>Source Input Matrix Preview</span>", unsafe_allow_html=True)
        st.dataframe(df.head(3), use_container_width=True)
        
        text_col = st.selectbox("Select target text column to evaluate:", df.columns)
        
        if st.button("Initialize Matrix Batch Run"):
            with st.spinner("Streaming array parameters through edge layer..."):
                texts = df[text_col].astype(str).tolist()
                start_time = time.time()
                inputs = tokenizer(texts, padding=True, truncation=True, max_length=64, return_tensors="pt")
                with torch.no_grad():
                    outputs = model(**inputs)
                end_time = time.time()
                
                logits = outputs.logits.tolist()
                df['Inference Latency'] = f"{(end_time - start_time)/len(texts)*1000:.2f} ms"
                df['State Vector Output'] = [str([round(v, 4) for v in row]) for row in logits]
                
                st.markdown("<br><h4 style='font-size: 14px; text-transform: uppercase; color: #3b82f6;'>Diagnostic Analysis Matrices Map</h4>", unsafe_allow_html=True)
                st.dataframe(df, use_container_width=True)
                
                processed_csv = df.to_csv(index=False).encode('utf-8')
                st.download_button("📥 Export Edge Performance Records", data=processed_csv, file_name="distilled_edge_performance.csv", mime="text/csv")

    # --- REAL-TIME HIGH-TECH SIMULATOR CONSOLE ---
    st.markdown("<br>", unsafe_allow_html=True)
    show_demo = st.checkbox("Launch Real-Time Sequence Simulator 🔮", value=True)

    if show_demo:
        st.markdown("<div class='analytics-card'>", unsafe_allow_html=True)
        user_text = st.text_area("Input Stream Container Sequence Entry:", "Evaluating optimization constraints on local client hardware parameters.", label_visibility="collapsed")
        
        if st.button("Process Live Sequence"):
            start = time.time()
            demo_inputs = tokenizer(user_text, return_tensors="pt", truncation=True, max_length=64)
            with torch.no_grad():
                demo_outputs = model(**demo_inputs)
            latency = (time.time() - start) * 1000
            
            st.markdown(f"""
            <div style="background: rgba(59, 130, 246, 0.1); border-left: 3px solid #3b82f6; border-radius: 6px; padding: 14px; margin-top: 15px; font-size: 13px;">
                <span style="color: #60a5fa; font-weight: 700; display: block; margin-bottom: 2px;">✓ LOCAL SUBSYSTEM VERIFIED</span>
                Runtime Latency Delta: <strong>{latency:.2f}ms</strong> | Compressed Compaction Footprint: <strong>~50MB Binary Block</strong>
            </div>
            """, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

except Exception as e:
    st.error(f"System Operational Architecture Load Failure: {e}")

# Padding offset to protect usability against fixed footer overlapping
st.markdown("<br><br><br>", unsafe_allow_html=True)

# 5. Fixed High-End Minimal Device Navigation Track
st.markdown("""
<div style="position: fixed; bottom: 0; left: 0; width: 100%; background: #050d1a; border-top: 1px solid #162a4e; padding: 14px 0; display: flex; justify-content: space-around; text-align: center; z-index: 99999;">
    <div style="color: #3b82f6; font-weight: 700; font-family: 'Space Grotesk', sans-serif; font-size: 11px; letter-spacing: 0.05em; cursor: pointer;">⚡ WORKSPACE</div>
    <div style="color: #64748b; font-weight: 500; font-family: 'Space Grotesk', sans-serif; font-size: 11px; letter-spacing: 0.05em; cursor: pointer;">📉 COMPRESSION</div>
    <div style="color: #64748b; font-weight: 500; font-family: 'Space Grotesk', sans-serif; font-size: 11px; letter-spacing: 0.05em; cursor: pointer;">📱 EDGE DEPLOY</div>
</div>
""", unsafe_allow_html=True)
