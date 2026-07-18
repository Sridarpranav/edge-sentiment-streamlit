import streamlit as st
from optimum.onnxruntime import ORTModelForSequenceClassification
from transformers import AutoTokenizer
import torch
import pandas as pd

# Set page config immediately
st.set_page_config(page_title="UQVision - ML Uncertainty Engine", page_icon="📊", layout="centered")

# CSS injected globally to handle styling without multiline indentation breaks
st.markdown("<style>button { border-radius: 8px !important; }</style>", unsafe_allow_value=True)

# Navbar layout structure elements
st.markdown('<div style="display:flex; justify-content:space-between; align-items:center; border-bottom:1px solid #e2e8f0; padding-bottom:10px; margin-bottom:20px;"><div><h2 style="margin:0; color:#1e3a8a;">UQVision</h2><span style="font-size:10px; color:#64748b;">ML UNCERTAINTY ENGINE</span></div><div style="font-size:12px; color:#64748b;">🌙 Dark Mode</div></div>', unsafe_allow_value=True)

# Main Hero Text Display
st.markdown('<div style="text-align:center; background-color:rgba(147,51,234,0.1); padding:5px; border-radius:20px; width:280px; margin:0 auto; color:#7c3aed; font-weight:bold; font-size:12px;">🛡️ Enterprise ML Confidence Engine</div>', unsafe_allow_value=True)
st.markdown('<h1 style="text-align:center; margin-top:10px;">Uncertainty Quantification<br><span style="color:#2563eb;">For Machine Learning</span></h1>', unsafe_allow_value=True)
st.markdown('<p style="text-align:center; color:#64748b; font-size:14px;">Don\'t rely on raw point predictions. Quantify text sequence structures with highly distilled, ultra-fast 8-bit quantized ONNX architectures.</p>', unsafe_allow_value=True)

# --- Model Loading Process ---
@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(".")
    model = ORTModelForSequenceClassification.from_pretrained(".", file_name="model_quantized.onnx")
    return tokenizer, model

try:
    tokenizer, model = load_model()
    
    # --- SECTION A: BATCH CSV FILE PROCESSOR ---
    st.subheader("📊 Batch Inference via CSV Upload")
    uploaded_file = st.file_uploader("Upload CSV Dataset", type=["csv"])

    if uploaded_file is not None:
        # Load the uploaded dataset
        df = pd.read_csv(uploaded_file)
        st.write("📋 **Uploaded Data Preview:**", df.head(3))
        
        # Look for a text column automatically
        text_column = st.selectbox("Select the text column to analyze:", df.columns)
        
        if st.button("🚀 Run Fast Batch Prediction"):
            with st.spinner("Processing batch predictions with ONNX..."):
                predictions = []
                confidences = []
                
                # Perform fast inference across rows
                for text_item in df[text_column].astype(str):
                    inputs = tokenizer(text_item, return_tensors="pt", truncation=True, max_length=64)
                    with torch.no_grad():
                        outputs = model(**inputs)
                    probs = torch.softmax(outputs.logits, dim=-1)[0]
                    pred = torch.argmax(probs).item()
                    conf = probs[pred].item() * 100
                    
                    predictions.append("Positive 🟢" if pred == 1 else "Negative 🔴")
                    confidences.append(f"{conf:.2f}%")
                
                # Append computed lists back to dataframe
                df['Sentiment Prediction'] = predictions
                df['Confidence Score'] = confidences
                
                st.success("✨ Processing Complete!")
                st.dataframe(df)
                
                # Provide a quick button to download processed data back
                csv_data = df.to_csv(index=False).encode('utf-8')
                st.download_button(label="📥 Download Analyzed CSV", data=csv_data, file_name="analyzed_predictions.csv", mime="text/csv")

    st.markdown("<hr>", unsafe_allow_value=True)

    # --- SECTION B: SINGLE TEXT INFERENCE ---
    st.subheader("✨ Real-Time Single Demo Inference")
    user_input = st.text_area("Type text sample:", "This application implementation runs lightning fast!")

    if st.button("Analyze Demo Text"):
        if user_input.strip() != "":
            inputs = tokenizer(user_input, return_tensors="pt", truncation=True, max_length=64)
            outputs = model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1)[0]
            
            prediction = torch.argmax(probs).item()
            confidence = probs[prediction].item() * 100
            
            label_map = {0: "Negative 🔴", 1: "Positive 🟢"}
            st.info(f"**Prediction:** {label_map[prediction]} | **Confidence:** {confidence:.2f}%")
        else:
            st.warning("Please input text characters to analyze.")

except Exception as e:
    st.error(f"Initialization issue: {e}")
