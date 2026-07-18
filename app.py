# 6. Premium Tech-Stack Specs Matrix at the Bottom
st.markdown("---")
st.markdown("<h3 style='font-size: 16px; letter-spacing: 0.05em; text-transform: uppercase; color: #64748b; margin-bottom: 20px;'>🛠️ Deployment & Environment Architecture</h3>", unsafe_allow_html=True)

# CSS injecting inline for custom layout styling of micro-pills
st.markdown("""
<style>
    .spec-card {
        background: rgba(11, 21, 40, 0.4) !important;
        border: 1px solid #162a4e !important;
        border-radius: 10px;
        padding: 16px;
        height: 100%;
    }
    .spec-header {
        font-size: 12px;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #3b82f6;
        margin-bottom: 12px;
        border-left: 2px solid #3b82f6;
        padding-left: 8px;
    }
    .tech-pill {
        display: inline-block;
        background: rgba(30, 41, 59, 0.5);
        border: 1px solid #334155;
        color: #e2e8f0;
        font-size: 11px;
        font-weight: 500;
        padding: 4px 10px;
        border-radius: 6px;
        margin: 4px 2px;
    }
</style>
""", unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    <div class="spec-card">
        <div class="spec-header">Target Environments</div>
        <div class="tech-pill">Google Colab Compute</div>
        <div class="tech-pill">VS Code Dev Environment</div>
        <div class="tech-pill">ARM64 Architecture</div>
        <div class="tech-pill">Raspberry Pi Runtime</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="spec-card">
        <div class="spec-header">Runtime Frameworks</div>
        <div class="tech-pill">PyTorch Core</div>
        <div class="tech-pill">Hugging Face Optimum</div>
        <div class="tech-pill">ONNX Runtime Engine</div>
        <div class="tech-pill">TensorFlow Lite</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="spec-card">
        <div class="spec-header">Benchmark Suites</div>
        <div class="tech-pill">SST-2 (Sentiment)</div>
        <div class="tech-pill">IMDB Evaluation Matrix</div>
        <div class="tech-pill">AG News Classification</div>
    </div>
    """, unsafe_allow_html=True)
