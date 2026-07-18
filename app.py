import streamlit as st
from optimum.onnxruntime import ORTModelForSequenceClassification
from transformers import AutoTokenizer
import torch
import pandas as pd

# 1. Page Configuration
st.set_page_config(
    page_title="EdgeDistill - Model Compression Engine",
    page_icon="⚡",
    layout="centered"
)

# 2. Modern Light Theme Inline CSS (Fixed unsafe_allow_html argument)
st.markdown("<style>.stApp { background-color: #f8fafc; color: #0f172a; } div[data-testid='stFileUploader'] { max-width: 100%; margin: 20px 0; background-color: #ffffff; border: 2px dashed #e2e8f0; border-radius: 12px; padding: 15px; } .stButton>button { background-color: #6d28d9 !important; color: white !important; width: 100%; border-radius: 8px; border: none; padding: 10px; font-weight: bold; box-shadow: 0 4px 6px -1px rgba(109, 40, 217, 0.2); } .stDataFrame { background-color: white; border-radius: 8px; }</style>", unsafe_allow_html=True)

# 3. Clean Product Header Custom Design
st.markdown('<div style="display: flex; justify-content: space-between; align-items: center; padding: 10px 0; border-bottom: 2px solid #e2e8f0; margin-bottom: 30px;"><div><h1 style="font-size: 24px; font-weight: 800; color: #0f172a; margin: 0;">Edge<span style="color: #6d28d9;">Distill</span></h1><div style="font-size: 11px; color: #64748b; letter-spacing: 1px; text-transform: uppercase; margin-top: -5px;">Small Language Model Distillation</div></div><div style="background: #ffffff; padding: 6px 16px; border-radius: 20px; font-size: 13px; border: 1px solid #e2e8f0; font-weight: 500; color: #475569;">💡 Light Mode</div></div>', unsafe_allow_html=True)

# 4. Hero & Project Identity Presentation
st.markdown('<div style="text-align: center; margin-top: 20px;"><span style="display: inline-block; background-color: rgba(109, 40, 217, 0.1); border: 1px solid rgba(109, 40, 217, 0.2); color: #6d28d9; padding: 6px 16px; border-radius: 20px; font-size: 13px; font-weight: 600;">📱 Edge AI & Hardware Optimization Engine</span></div>', unsafe_allow_html=True)
st.markdown('<h2 style="text-align: center; font-size: 36px; font-weight: 800; line-height: 1.2; margin-top: 25px; margin-bottom: 15px; color: #0f172a;">Edge Sentiment Classifier<br><span style="background: linear-gradient(45deg, #6d28d9, #3b82f6); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">Distilled Student Inference</span></h2>', unsafe_allow_html=True)
st.markdown('<p style="text-align: center; font-size: 16px; color: #475569; line-height: 1.6; max-width: 580px; margin: 0 auto 40px auto;">Running a highly compressed, <strong>Int8 Quantized ONNX Model</strong> directly in the browser environment. Optimized for zero-latency deployment on mobile phones and Raspberry Pi setups.</p>', unsafe_allow_html=True)

# 5. Core Optimized Model Loader
@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(".")
    model = ORTModelForSequenceClassification.from_pretrained(".", file_name="model_quantized.onnx")
    return tokenizer, model

try:
    tokenizer, model = load_model()

    # --- CSV UPLOADER FEATURE ---
    st.markdown("### 📊 Batch Evaluation Panel")
    uploaded_file = st.file_uploader("Upload CSV Dataset", type=["csv"])

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.markdown("#### 📋 Dataset Preview")
        st.dataframe(df.head(5))
        
        text_col = st.selectbox("Select target text column for pipeline processing:", df.columns)
        
        if st.button("⚡ Run High-Speed Batch Prediction"):
            with st.spinner("Processing rows using lightning-fast ONNX batching..."):
                texts = df[text_col].astype(str).tolist()
                
                inputs = tokenizer(texts, padding=True, truncation=True, max_length=64, return_tensors="pt")
                
                with torch.no_grad():
                    outputs = model(**inputs)
                    
                probs = torch.softmax(outputs.logits, dim=-1)
                preds = torch.argmax(probs, dim=-1).tolist()
                confidences = (torch.max(probs, dim=-1).values * 100).tolist()
                
                df['Distilled Prediction'] = ["Positive 🟢" if p == 1 else "Negative 🔴" for p in preds]
                df['Confidence Score'] = [f"{c:.2f}%" for c in confidences]
                
                st.markdown("#### ✨ Processed Results")
                st.dataframe(df)
                
                processed_csv = df.to_csv(index=False).encode('utf-8')
                st.download_button("📥 Download Processed Dataset", data=processed_csv, file_name="edge_distilled_results.csv", mime="text/csv")

    # --- LIVE INFERENCE DEMO TRIGGER ---
    st.markdown("<br>", unsafe_allow_html=True)
    show_demo = st.checkbox("🔮 Show Real-time Demo Input Panel", value=True)

    if show_demo:
        st.markdown("<div style='background-color: white; padding: 20px; border-radius: 12px; border: 1px solid #e2e8f0; margin-bottom: 20px;'>", unsafe_allow_html=True)
        user_text = st.text_area("Live Text Input:", "This optimized student model process runs exceptionally fast on low-powered edge devices!")
        if st.button("Run Edge Inference"):
            demo_inputs = tokenizer(user_text, return_tensors="pt", truncation=True, max_length=64)
            with torch.no_grad():
                demo_outputs = model(**demo_inputs)
            demo_probs = torch.softmax(demo_outputs.logits, dim=-1)[0]
            demo_pred = torch.argmax(demo_probs).item()
            demo_conf = demo_probs[demo_pred].item() * 100
            
            res_label = "Positive 🟢" if demo_pred == 1 else "Negative 🔴"
            st.success(f"**Result:** {res_label} | **Confidence:** {demo_conf:.2f}%")
        st.markdown("</div>", unsafe_allow_html=True)

except Exception as e:
    st.error(f"Error executing engine backend architecture: {e}")

# 6. Extra Tech-Stack Specs Features Added Effortlessly at the Bottom
st.markdown("---")
st.markdown("### 🛠️ Architecture Stack & Distillation Blueprint")

col1, col2, col3 = st.columns(3)
with col1:
    st.markdown("**📂 Target Benchmarks**\n* SST-2\n* IMDB Reviews\n* AG News")
with col2:
    st.markdown("**🔧 Optimizations**\n* Knowledge Distillation\n* Int8 Quantization\n* Structural Pruning")
with col3:
    st.markdown("**📦 Engine Libraries**\n* PyTorch & Transformers\n* ONNX Runtime\n* TensorFlow Lite")

# Bottom Navigation Bar Mockup styled cleanly for Light Mode
st.markdown('<div style="position: fixed; bottom: 0; left: 0; width: 100%; background-color: #ffffff; border-top: 1px solid #e2e8f0; padding: 12px 0; display: flex; justify-content: space-around; text-align: center; z-index: 999;"><div style="color: #6d28d9; font-weight: bold; font-size: 12px; cursor: pointer;">⚡<br>Inference</div><div style="color: #64748b; font-size: 12px; cursor: pointer;">📊<br>Batch Process</div><div style="color: #64748b; font-size: 12px; cursor: pointer;">🛠️<br>Specs Blueprint</div></div>', unsafe_allow_html=True)
