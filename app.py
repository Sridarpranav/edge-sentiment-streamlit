import streamlit as st
from optimum.onnxruntime import ORTModelForSequenceClassification
from transformers import AutoTokenizer
import torch
import pandas as pd
import time

# 1. Page Configuration
st.set_page_config(
    page_title="NSRLM LAB",
    page_icon="⚡",
    layout="centered"
)

# 2. Premium High-Contrast Typography & Dynamic Layout Rules
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;700&family=Inter:wght@400;500;600&display=swap');

    /* Global Typography overrides */
    h1, h2, h3, h4, .premium-header {
        font-family: 'Space Grotesk', sans-serif !important;
        letter-spacing: -0.03em !important;
    }
    p, span, label, input, .stMarkdown {
        font-family: 'Inter', sans-serif !important;
    }

    /* Light Mode Variables & Structuring */
    @media (prefers-color-scheme: light) {
        .stApp {
            background-color: #f9f6f0 !important; /* Premium Editorial Cream */
            color: #111827 !important;
        }
        .dashboard-card {
            background-color: #ffffff !important;
            border: 1px solid #e5e7eb !important;
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.03);
        }
        .metric-label { color: #6b7280 !important; }
        .bottom-bar { background-color: #ffffff !important; border-top: 1px solid #e5e7eb; }
    }

    /* Dark Mode Variables & Structuring */
    @media (prefers-color-scheme: dark) {
        .stApp {
            background-color: #0d0f0e !important; /* Luxury Deep Obsidian */
            color: #f9fafb !important;
        }
        .dashboard-card {
            background-color: #161917 !important;
            border: 1px solid #242926 !important;
            box-shadow: 0 4px 25px rgba(0, 0, 0, 0.4);
        }
        .metric-label { color: #9ca3af !important; }
        .bottom-bar { background-color: #161917 !important; border-top: 1px solid #242926; }
    }

    /* Unified Layout Blocks */
    .dashboard-card {
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 20px;
    }

    /* Ultra-Clean High-Contrast Action Buttons */
    .stButton>button { 
        background-color: #10b981 !important; 
        color: white !important; 
        width: 100%; 
        border-radius: 8px; 
        border: none; 
        padding: 14px; 
        font-size: 14px;
        font-weight: 600; 
        letter-spacing: 0.03em;
        text-transform: uppercase;
        transition: opacity 0.2s ease;
    }
    .stButton>button:hover {
        opacity: 0.9;
    }

    /* Clean up the file uploader styling */
    div[data-testid='stFileUploader'] {
        border-radius: 12px;
        padding: 15px;
    }
</style>
""", unsafe_allow_html=True)

# 3. Minimal Editorial Navbar Header
st.markdown("""
<div style="display: flex; justify-content: space-between; align-items: center; padding: 15px 0; margin-bottom: 30px;">
    <div>
        <h1 style="font-size: 18px; font-weight: 700; margin: 0; letter-spacing: 0.05em;">NSRLM <span style="color: #10b981;">LAB</span></h1>
    </div>
    <div style="font-size: 10px; letter-spacing: 1px; text-transform: uppercase; font-weight: 700; color: #10b981;">
        01 / WORKSPACE
    </div>
</div>
""", unsafe_allow_html=True)

# 4. Premium Headline Block
st.markdown('<h2 style="font-size: 34px; font-weight: 700; line-height: 1.15; margin-bottom: 8px;">Find the metrics making your edge devices unstable.</h2>', unsafe_allow_html=True)
st.markdown('<p style="font-size: 15px; line-height: 1.5; opacity: 0.8; margin-bottom: 30px;">Evaluate optimization footprints and inference speed constraints instantly across mobile hardware configurations.</p>', unsafe_allow_html=True)

# 5. Core Optimized Model Loader Execution
@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(".")
    model = ORTModelForSequenceClassification.from_pretrained(".", file_name="model_quantized.onnx")
    return tokenizer, model

try:
    tokenizer, model = load_model()

    # --- LUXURY DIGITAL HARDWARE DIALS ---
    st.markdown('<div class="dashboard-card">', unsafe_allow_html=True)
    m1, m2, m3 = st.columns(3)
    with m1:
        st.markdown('<span style="font-size: 11px; text-transform: uppercase;" class="metric-label">Large AI Size</span>', unsafe_allow_html=True)
        st.markdown('<h3 style="margin: 5px 0 0 0; font-size: 24px; font-weight: 700;">400 MB</h3>', unsafe_allow_html=True)
    with m2:
        st.markdown('<span style="font-size: 11px; text-transform: uppercase;" class="metric-label">Small SLM Size</span>', unsafe_allow_html=True)
        st.markdown('<h3 style="margin: 5px 0 0 0; font-size: 24px; font-weight: 700; color: #10b981;">50 MB</h3>', unsafe_allow_html=True)
    with m3:
        st.markdown('<span style="font-size: 11px; text-transform: uppercase;" class="metric-label">Target Latency</span>', unsafe_allow_html=True)
        st.markdown('<h3 style="margin: 5px 0 0 0; font-size: 24px; font-weight: 700;">&lt; 15ms</h3>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # --- ADVANCED HIGH-CLASS FILE DROPS ---
    st.markdown('<h3 style="font-size: 20px; font-weight: 700; margin-bottom: 10px;">Drop in a table. Get a diagnosis.</h3>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Upload CSV Matrix", type=["csv"], label_visibility="collapsed")

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.markdown("<br><span style='font-size: 12px; font-weight:700; text-transform:uppercase;'>Dataset View</span>", unsafe_allow_html=True)
        st.dataframe(df.head(3), use_container_width=True)
        
        text_col = st.selectbox("Specify target evaluation column", df.columns)
        
        if st.button("Run Diagnostics Pipeline"):
            with st.spinner("Processing token array configurations across edge compilation parameters..."):
                texts = df[text_col].astype(str).tolist()
                
                start_time = time.time()
                inputs = tokenizer(texts, padding=True, truncation=True, max_length=64, return_tensors="pt")
                
                with torch.no_grad():
                    outputs = model(**inputs)
                end_time = time.time()
                
                logits = outputs.logits.tolist()
                df['Latency Target'] = f"{(end_time - start_time)/len(texts)*1000:.2f} ms"
                df['State Vector Output'] = [str([round(v, 4) for v in row]) for row in logits]
                
                st.markdown("<br><span style='font-size: 12px; font-weight:700; text-transform:uppercase;'>Diagnostic Board Output</span>", unsafe_allow_html=True)
                st.dataframe(df, use_container_width=True)
                
                processed_csv = df.to_csv(index=False).encode('utf-8')
                st.markdown("<br>", unsafe_allow_html=True)
                st.download_button("📥 Export Performance Logs", data=processed_csv, file_name="nsrlm_edge_analytics.csv", mime="text/csv")

    # --- PREMIUM MOBILE DEMO BENCHMARK PANEL ---
    st.markdown("<br>", unsafe_allow_html=True)
    show_demo = st.checkbox("🔮 Launch Real-Time Sequence Simulator", value=True)

    if show_demo:
        st.markdown("<div class='dashboard-card'>", unsafe_allow_html=True)
        user_text = st.text_area("Live Input Text Container:", "Simulating processing constraints within modern mobile processor sandboxes.")
        if st.button("Process Live Sequence"):
            start = time.time()
            demo_inputs = tokenizer(user_text, return_tensors="pt", truncation=True, max_length=64)
            with torch.no_grad():
                demo_outputs = model(**demo_inputs)
            latency = (time.time() - start) * 1000
            
            st.markdown(f"""
            <div style="background-color: rgba(16, 185, 129, 0.1); border-left: 3px solid #10b981; border-radius: 4px; padding: 15px; margin-top: 15px;">
                <div style="color: #10b981; font-weight: 700; font-family: 'Space Grotesk', sans-serif; font-size: 11px; letter-spacing: 0.05em; margin-bottom: 2px;">✓ LOCAL COMPILER RUN SUCCESSFUL</div>
                <div style="font-size: 13px;">Binary Footprint: <strong>~50 MB Target</strong> | Latency Metric: <strong>{latency:.2f}ms</strong></div>
            </div>
            """, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

except Exception as e:
    st.error(f"Error executing engine backend architecture: {e}")

# Padding spacer for sticky bottom navigation layout compatibility
st.markdown("<br><br><br><br>", unsafe_allow_html=True)

# 6. Bespoke Fixed Ultra-Premium Mobile Bottom Navigation Track
st.markdown("""
<div class="bottom-bar" style="position: fixed; bottom: 0; left: 0; width: 100%; padding: 14px 0; display: flex; justify-content: space-around; text-align: center; z-index: 999999;">
    <div style="color: #10b981; font-weight: 700; font-family: 'Space Grotesk', sans-serif; font-size: 11px; letter-spacing: 0.05em; cursor: pointer;">
        ⚡<br>WORKSPACE
    </div>
    <div style="opacity: 0.5; font-weight: 500; font-family: 'Space Grotesk', sans-serif; font-size: 11px; letter-spacing: 0.05em; cursor: pointer;">
        📊<br>ANALYSIS
    </div>
    <div style="opacity: 0.5; font-weight: 500; font-family: 'Space Grotesk', sans-serif; font-size: 11px; letter-spacing: 0.05em; cursor: pointer;">
        🌿<br>METHOD
    </div>
</div>
""", unsafe_allow_html=True)
