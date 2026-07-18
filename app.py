import streamlit as st
from optimum.onnxruntime import ORTModelForSequenceClassification
from transformers import AutoTokenizer
import torch

# --- Page Config & Styling Layout ---
st.set_page_config(page_title="UQVision - ML Uncertainty Engine", page_icon="📊", layout="centered")

# Inject Custom CSS to build the dark theme dashboard layout
st.markdown("""
<style>
    /* Main container styling background */
    .stApp {
        background-color: #0d1224;
        color: #ffffff;
    }
    
    /* Top Header Bar styling matching reference mockup */
    .uq-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 10px 0px;
        border-bottom: 1px solid #1e293b;
        margin-bottom: 30px;
    }
    .uq-logo-text {
        font-size: 24px;
        font-weight: 800;
        color: #ffffff;
        margin: 0;
    }
    .uq-logo-text span {
        color: #00bcd4;
    }
    .uq-subtitle {
        font-size: 11px;
        color: #94a3b8;
        letter-spacing: 1px;
        text-transform: uppercase;
        margin-top: -5px;
    }
    
    /* Enterprise Pill Badge styling */
    .badge-container {
        text-align: center;
        margin-top: 20px;
    }
    .enterprise-badge {
        display: inline-block;
        background-color: rgba(147, 51, 234, 0.15);
        border: 1px solid rgba(147, 51, 234, 0.3);
        color: #c084fc;
        padding: 6px 16px;
        border-radius: 20px;
        font-size: 13px;
        font-weight: 600;
        letter-spacing: 0.5px;
    }
    
    /* Premium Gradient Hero Headings styling */
    .hero-title {
        text-align: center;
        font-size: 38px;
        font-weight: 800;
        line-height: 1.2;
        margin-top: 25px;
        margin-bottom: 15px;
        color: #ffffff;
    }
    .hero-gradient {
        background: linear-gradient(45deg, #a855f7, #3b82f6, #06b6d4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    /* Narrative Description typography */
    .hero-desc {
        text-align: center;
        font-size: 16px;
        color: #94a3b8;
        line-height: 1.6;
        max-width: 550px;
        margin: 0 auto 40px auto;
    }
    .hero-desc strong {
        color: #e2e8f0;
    }

    /* Target styling elements for native streamlit file uploader wrapper container */
    div[data-testid="stFileUploader"] {
        max-width: 400px;
        margin: 0 auto;
        background-color: #7c3aed !important;
        border-radius: 12px;
        padding: 5px;
    }
    div[data-testid="stFileUploader"] section {
        background-color: #7c3aed !important;
        color: white !important;
        border: none !important;
    }
    div[data-testid="stFileUploader"] label {
        color: white !important;
        font-weight: bold !important;
        text-align: center;
    }
    
    /* Text input text-area structural adjustments */
    .stTextArea textarea {
        background-color: #1e293b !important;
        color: #ffffff !important;
        border: 1px solid #334155 !important;
        border-radius: 8px !important;
    }
</style>
""", unsafe_allow_value=True)

# --- Top Dashboard Navbar Structure ---
st.markdown("""
<div class="uq-header">
    <div>
        <h1 class="uq-logo-text">UQ<span>Vision</span></h1>
        <div class="uq-subtitle">ML UNCERTAINTY ENGINE</div>
    </div>
    <div style="background: rgba(255,255,255,0.05); padding: 6px 16px; border-radius: 20px; font-size: 13px; border: 1px solid rgba(255,255,255,0.1);">
        ☀️ Light / 🌙 Dark
    </div>
</div>
""", unsafe_allow_value=True)

# --- Central Hero Banner Block Layout ---
st.markdown("""
<div class="badge-container">
    <span class="enterprise-badge">🛡️ Enterprise ML Confidence & Interval Bounds</span>
</div>
<h2 class="hero-title">
    Uncertainty Quantification<br>
    <span class="hero-gradient">For Machine Learning</span>
</h2>
<p class="hero-desc">
    Don't rely on raw point predictions. Quantify <strong>Aleatoric</strong> (data noise) and 
    <strong>Epistemic</strong> (model ignorance) uncertainty with Split Conformal Prediction & Bootstrap Ensembles.
</p>
""", unsafe_allow_value=True)

# --- Interactive Application Actions Pipeline ---
# Primary CSV Data Upload Control element block wrapper mapping
uploaded_file = st.file_uploader("Upload CSV Dataset", type=["csv"], label_visibility="collapsed")

st.markdown("<br><h4 style='text-align: center; color: #64748b;'>— OR TRY REAL-TIME EDGE INFERENCE —</h4>", unsafe_allow_value=True)

# --- Active Model Execution Interface Section ---
@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(".")
    model = ORTModelForSequenceClassification.from_pretrained(".", file_name="model_quantized.onnx")
    return tokenizer, model

try:
    tokenizer, model = load_model()
    
    user_input = st.text_area("Enter textual sequence for analysis:", "This deployment implementation is incredibly responsive!", label_visibility="collapsed")

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        analyze_btn = st.button("✨ Try Live Demo Inference", use_container_width=True)

    if analyze_btn:
        if user_input.strip() != "":
            inputs = tokenizer(user_input, return_tensors="pt", truncation=True, max_length=64)
            outputs = model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1)[0]
            
            prediction = torch.argmax(probs).item()
            confidence = probs[prediction].item() * 100
            
            label_map = {0: "Negative Classification 🔴", 1: "Positive Classification 🟢"}
            
            # Formatted outputs dashboard card UI element presentation frame
            st.markdown(f"""
            <div style="background-color: #1e293b; border-left: 4px solid #06b6d4; padding: 20px; border-radius: 8px; margin-top: 20px;">
                <h5 style="margin: 0 0 10px 0; color: #94a3b8; text-transform: uppercase; font-size: 12px; letter-spacing: 1px;">Engine Core Inference Result</h5>
                <p style="margin: 0; font-size: 18px; font-weight: bold; color: #ffffff;">Prediction Target: {label_map[prediction]}</p>
                <p style="margin: 5px 0 0 0; font-size: 14px; color: #38bdf8;">Calculated Model Confidence Set: {confidence:.2f}%</p>
            </div>
            """, unsafe_allow_value=True)
        else:
            st.warning("Please submit input characters within the entry prompt domain context.")
            
except Exception as e:
    st.error(f"Error accessing model binary sequences: {e}")

# --- Sticky Bottom Navigation Bar layout structure emulation ---
st.markdown("""
<br><br><br>
<div style="position: fixed; bottom: 0; left: 0; width: 100%; background-color: #090d16; border-top: 1px solid #1e293b; padding: 12px 0; display: flex; justify-content: space-around; text-align: center; z-index: 999;">
    <div style="color: #a855f7; font-weight: bold; font-size: 12px; cursor: pointer;">🏠<br>Home</div>
    <div style="color: #64748b; font-size: 12px; cursor: pointer;">📤<br>Upload</div>
    <div style="color: #64748b; font-size: 12px; cursor: pointer;">📊<br>Dashboard</div>
</div>
""", unsafe_allow_value=True)
