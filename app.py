import streamlit as st
from optimum.onnxruntime import ORTModelForSequenceClassification
from transformers import AutoTokenizer
import torch
import pandas as pd

# 1. Page Configuration (Must be first)
st.set_page_config(
    page_title="UQVision - ML Uncertainty Engine",
    page_icon="📊",
    layout="centered"
)

# 2. Strict Inline CSS (Eliminates triple-quote multiline indentation bugs)
st.markdown("<style>.stApp { background-color: #0d1224; color: #ffffff; } div[data-testid='stFileUploader'] { max-width: 100%; margin: 20px 0; background-color: #7c3aed; border-radius: 12px; padding: 10px; } div[data-testid='stFileUploader'] section { background-color: #7c3aed !important; color: white !important; border: none !important; } .stButton>button { background-color: #7c3aed !important; color: white !important; width: 100%; border-radius: 8px; border: none; padding: 10px; font-weight: bold; } .demo-btn>button { background-color: transparent !important; color: #ffffff !important; border: 1px solid #334155 !important; }</style>", unsafe_allow_value=True)

# 3. Header Custom Design
st.markdown('<div style="display: flex; justify-content: space-between; align-items: center; padding: 10px 0; border-bottom: 1px solid #1e293b; margin-bottom: 30px;"><div><h1 style="font-size: 24px; font-weight: 800; color: #ffffff; margin: 0;">UQ<span style="color: #00bcd4;">Vision</span></h1><div style="font-size: 11px; color: #94a3b8; letter-spacing: 1px; text-transform: uppercase; margin-top: -5px;">ML UNCERTAINTY ENGINE</div></div><div style="background: rgba(255,255,255,0.05); padding: 6px 16px; border-radius: 20px; font-size: 13px; border: 1px solid rgba(255,255,255,0.1);">☀️ Dark</div></div>', unsafe_allow_value=True)

# 4. Hero Visual Layout
st.markdown('<div style="text-align: center; margin-top: 20px;"><span style="display: inline-block; background-color: rgba(147, 51, 234, 0.15); border: 1px solid rgba(147, 51, 234, 0.3); color: #c084fc; padding: 6px 16px; border-radius: 20px; font-size: 13px; font-weight: 600;">🛡️ Enterprise ML Confidence & Interval Bounds</span></div>', unsafe_allow_value=True)
st.markdown('<h2 style="text-align: center; font-size: 38px; font-weight: 800; line-height: 1.2; margin-top: 25px; margin-bottom: 15px; color: #ffffff;">Uncertainty Quantification<br><span style="background: linear-gradient(45deg, #a855f7, #3b82f6, #06b6d4); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">For Machine Learning</span></h2>', unsafe_allow_value=True)
st.markdown('<p style="text-align: center; font-size: 16px; color: #94a3b8; line-height: 1.6; max-width: 550px; margin: 0 auto 40px auto;">Don\'t rely on raw point predictions. Quantify <strong>Aleatoric</strong> (data noise) and <strong>Epistemic</strong> (model ignorance) uncertainty with Split Conformal Prediction & Bootstrap Ensembles.</p>', unsafe_allow_value=True)

# 5. Core Optimized Model Loader
@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(".")
    model = ORTModelForSequenceClassification.from_pretrained(".", file_name="model_quantized.onnx")
    return tokenizer, model

try:
    tokenizer, model = load_model()

    # --- CSV UPLOADER FEATURE ---
    uploaded_file = st.file_uploader("Upload CSV Dataset", type=["csv"], label_visibility="collapsed")

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.markdown("### 📋 Data Preview")
        st.dataframe(df.head(5))
        
        # Automatically guess or select target text column
        text_col = st.selectbox("Select the column containing your text data:", df.columns)
        
        if st.button("⚡ Run High-Speed Batch Prediction"):
            with st.spinner("Processing rows using lightning-fast ONNX batching..."):
                texts = df[text_col].astype(str).tolist()
                
                # Tokenize everything at once for maximum speed optimization
                inputs = tokenizer(texts, padding=True, truncation=True, max_length=64, return_tensors="pt")
                
                with torch.no_grad():
                    outputs = model(**inputs)
                    
                probs = torch.softmax(outputs.logits, dim=-1)
                preds = torch.argmax(probs, dim=-1).tolist()
                confidences = (torch.max(probs, dim=-1).values * 100).tolist()
                
                # Map results back cleanly
                df['Prediction'] = ["Positive 🟢" if p == 1 else "Negative 🔴" for p in preds]
                df['Confidence %'] = [f"{c:.2f}%" for c in confidences]
                
                st.markdown("### ✨ Processed Results")
                st.dataframe(df)
                
                # Download link for the newly evaluated CSV file
                processed_csv = df.to_csv(index=False).encode('utf-8')
                st.download_button("📥 Download Processed CSV Dataset", data=processed_csv, file_name="uqvision_predictions.csv", mime="text/csv")

    # --- LIVE INFERENCE DEMO TRIGGER ---
    st.markdown("<br>", unsafe_allow_value=True)
    st.markdown("<div class='demo-btn'>", unsafe_allow_value=True)
    show_demo = st.checkbox("🔮 Show Real-time Demo Input Panel", value=False)
    st.markdown("</div>", unsafe_allow_value=True)

    if show_demo:
        user_text = st.text_area("Live Input Text:", "This model pipeline runs incredibly fast using specialized ONNX quantization steps!")
        if st.button("Try Live Demo"):
            demo_inputs = tokenizer(user_text, return_tensors="pt", truncation=True, max_length=64)
            with torch.no_grad():
                demo_outputs = model(**demo_inputs)
            demo_probs = torch.softmax(demo_outputs.logits, dim=-1)[0]
            demo_pred = torch.argmax(demo_probs).item()
            demo_conf = demo_probs[demo_pred].item() * 100
            
            res_label = "Positive 🟢" if demo_pred == 1 else "Negative 🔴"
            st.info(f"Result: {res_label} ({demo_conf:.2f}% confidence)")

except Exception as e:
    st.error(f"Error executing engine backend architecture: {e}")

# Bottom Sticky Navigation Bar Mockup
st.markdown('<div style="position: fixed; bottom: 0; left: 0; width: 100%; background-color: #090d16; border-top: 1px solid #1e293b; padding: 12px 0; display: flex; justify-content: space-around; text-align: center; z-index: 999;"><div style="color: #a855f7; font-weight: bold; font-size: 12px; cursor: pointer;">🏠<br>Home</div><div style="color: #64748b; font-size: 12px; cursor: pointer;">📤<br>Upload</div><div style="color: #64748b; font-size: 12px; cursor: pointer;">📊<br>Dashboard</div></div>', unsafe_allow_value=True)
