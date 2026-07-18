import streamlit as st
from optimum.onnxruntime import ORTModelForSequenceClassification
from transformers import AutoTokenizer
import torch
import os

st.set_page_config(page_title="Edge Sentiment Classifier", page_icon="🚀")
st.title("🚀 Fast Edge Sentiment Analyzer")
st.write("This app runs an optimized, Int8 quantized ONNX model directly inside Streamlit Cloud.")

# Load the model files locally from the repo
@st.cache_resource
def load_model():
    # Looks for files in the current root directory
    tokenizer = AutoTokenizer.from_pretrained(".")
    model = ORTModelForSequenceClassification.from_pretrained(".", file_name="model_quantized.onnx")
    return tokenizer, model

try:
    tokenizer, model = load_model()
    
    user_input = st.text_area("Enter text to analyze sentiment:", "This streamlined approach is perfect!")

    if st.button("Analyze Sentiment"):
        if user_input.strip() != "":
            with st.spinner("Processing text on edge instance..."):
                inputs = tokenizer(user_input, return_tensors="pt", truncation=True, max_length=64)
                outputs = model(**inputs)
                probs = torch.softmax(outputs.logits, dim=-1)[0]
                
                prediction = torch.argmax(probs).item()
                confidence = probs[prediction].item() * 100
                
                label_map = {0: "Negative 🔴", 1: "Positive 🟢"}
                st.subheader("Result:")
                st.info(f"Prediction: **{label_map[prediction]}** (Confidence: {confidence:.2f}%)")
        else:
            st.warning("Please enter some text.")
            
except Exception as e:
    st.error(f"Error loading model files: {e}")
    st.info("Make sure model_quantized.onnx, config.json, and tokenizer files are uploaded to the root of this repository.")
