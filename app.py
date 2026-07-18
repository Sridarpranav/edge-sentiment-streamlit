# 6. Premium Tech-Stack Specs Matrix at the Bottom
st.markdown("---")
st.markdown("<h3 style='font-size: 13px; letter-spacing: 0.08em; text-transform: uppercase; color: #475569; margin-bottom: 20px; font-weight: 600;'>System Deployment & Environment Matrix</h3>", unsafe_allow_html=True)

# CSS injecting inline for ultra-premium dark terminal/cyber design
st.markdown("""
<style>
    .spec-card {
        background: rgba(8, 18, 36, 0.6) !important;
        border: 1px solid #1e293b !important;
        border-radius: 6px;
        padding: 18px;
        height: 100%;
        box-shadow: inset 0 1px 2px rgba(255, 255, 255, 0.05);
    }
    
    .spec-header {
        font-size: 11px;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        margin-bottom: 16px;
        display: flex;
        align-items: center;
    }
    
    /* Modern geometric replacement for emojis */
    .spec-header::before {
        content: '';
        display: inline-block;
        width: 4px;
        height: 12px;
        margin-right: 8px;
    }
    
    /* Variable tech coloring for distinct columns */
    .blue-matrix::before { background-color: #3b82f6; }
    .blue-matrix { color: #3b82f6; }
    
    .emerald-matrix::before { background-color: #10b981; }
    .emerald-matrix { color: #10b981; }
    
    .cyan-matrix::before { background-color: #06b6d4; }
    .cyan-matrix { color: #06b6d4; }

    .tech-pill {
        display: inline-block;
        background: rgba(15, 23, 42, 0.8);
        border: 1px solid #334155;
        color: #94a3b8;
        font-size: 11px;
        font-family: monospace;
        font-weight: 500;
        padding: 5px 10px;
        border-radius: 4px;
        margin: 4px 2px;
        transition: all 0.2s ease;
    }
    
    .tech-pill:hover {
        border-color: #475569;
        color: #f1f5f9;
        background: rgba(30, 41, 59, 0.5);
    }
</style>
""", unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    <div class="spec-card">
        <div class="spec-header blue-matrix">Target Infrastructure</div>
        <div class="tech-pill">Google Colab Compute</div>
        <div class="tech-pill">VS Code Remote Host</div>
        <div class="tech-pill">ARM64 Architecture</div>
        <div class="tech-pill">Raspberry Pi Local</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="spec-card">
        <div class="spec-header emerald-matrix">Runtime Frameworks</div>
        <div class="tech-pill">PyTorch Core</div>
        <div class="tech-pill">HF Optimum Engine</div>
        <div class="tech-pill">ONNX Runtime</div>
        <div class="tech-pill">TensorFlow Lite</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="spec-card">
        <div class="spec-header cyan-matrix">Validation Datasets</div>
        <div class="tech-pill">SST-2 Matrix</div>
        <div class="tech-pill">IMDB Evaluation</div>
        <div class="tech-pill">AG News Vector</div>
    </div>
    """, unsafe_allow_html=True)
