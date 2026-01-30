"""
MedGuard AI - Enhanced Dashboard
Multi-agent AI system for healthcare data quality and fairness
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import sys
import os
import ollama

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from utils.phi_scanner import PHIScanner, PHIDetectedError
from utils.data_quality import DataQualityAnalyzer
from utils.bias_detection import BiasDetector
from utils.synthetic_generator import SyntheticDataGenerator
from agents.medical_advisor_v2 import MedicalAdvisorV2
from agents.fairness_specialist import FairnessSpecialist
from agents.deployment_strategist import DeploymentStrategist

# ============================================================================
# PAGE CONFIG
# ============================================================================

st.set_page_config(
    page_title="MedGuard AI",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced Custom CSS
st.markdown("""
<style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&display=swap');
    
    /* Global Styles */
    * {
        font-family: 'Inter', sans-serif;
    }
    
    /* Remove ALL unnecessary padding */
    .block-container {
        padding-top: 2rem !important;
        padding-bottom: 0.3rem !important;
        padding-left: 1rem !important;
        padding-right: 1rem !important;
        max-width: 100% !important;
    }
    
    /* Remove padding from main content area */
    .main .block-container {
        padding-top: 2rem !important;
    }
    
    /* Tighten up all elements - NO margin/padding by default */
    .element-container {
        margin: 0 !important;
        padding: 0 !important;
    }
    
    /* Remove spacing from stMarkdown */
    .stMarkdown {
        margin-bottom: 0 !important;
        margin-top: 0 !important;
    }
    
    /* Main Header with Gradient Animation */
    .main-header {
        font-size: 3.5rem;
        font-weight: 800;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 25%, #f093fb 50%, #4facfe 75%, #00f2fe 100%);
        background-size: 300% 300%;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        animation: gradientShift 8s ease infinite;
        margin: 0.3rem 0 !important;
        padding: 0 !important;
        text-align: center;
        letter-spacing: -2px;
        line-height: 1.2;
    }
    
    @keyframes gradientShift {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    
    .subheader {
        font-size: 1.2rem;
        color: #a0a0a0;
        margin: 0.2rem 0 !important;
        padding: 0 !important;
        text-align: center;
        font-weight: 400;
        letter-spacing: 0.5px;
    }
    
    .badge-container {
        display: flex;
        justify-content: center;
        gap: 0.8rem;
        margin: 0.4rem 0 0.5rem 0 !important;
        padding: 0 !important;
        flex-wrap: wrap;
    }
    
    .badge {
        background: linear-gradient(135deg, #667eea22, #764ba222);
        border: 1px solid #667eea44;
        padding: 0.4rem 1rem;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 600;
        backdrop-filter: blur(10px);
        transition: all 0.3s ease;
    }
    
    .badge:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(102, 126, 234, 0.3);
    }
    
    /* Enhanced Metrics */
    [data-testid="stMetricValue"] {
        font-size: 2rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea, #764ba2);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    /* Card Styling - MINIMAL padding */
    .stAlert {
        border-radius: 10px;
        border: none;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        backdrop-filter: blur(10px);
        margin: 0.3rem 0 !important;
        padding: 0.6rem !important;
    }
    
    div[data-testid="stExpander"] {
        background: linear-gradient(135deg, #ffffff05, #ffffff08);
        border: 1px solid #ffffff15;
        border-radius: 10px;
        margin: 0.2rem 0 !important;
        padding: 0 !important;
        transition: all 0.3s ease;
    }
    
    div[data-testid="stExpander"]:hover {
        border-color: #667eea55;
        box-shadow: 0 5px 15px rgba(102, 126, 234, 0.2);
    }
    
    /* Tabs Enhancement */
    .stTabs [data-baseweb="tab-list"] {
        gap: 1rem;
        background: linear-gradient(135deg, #ffffff05, #ffffff08);
        padding: 0.5rem;
        border-radius: 15px;
        border: 1px solid #ffffff15;
    }
    
    .stTabs [data-baseweb="tab"] {
        border-radius: 10px;
        padding: 1rem 2rem;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea, #764ba2);
        box-shadow: 0 4px 10px rgba(102, 126, 234, 0.3);
    }
    
    /* Button Enhancements */
    .stButton > button {
        border-radius: 10px;
        font-weight: 600;
        padding: 0.75rem 2rem;
        border: none;
        background: linear-gradient(135deg, #667eea, #764ba2);
        color: white;
        transition: all 0.3s ease;
        box-shadow: 0 4px 10px rgba(102, 126, 234, 0.3);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.4);
    }
    
    /* Sidebar Enhancement */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0e1117 0%, #1a1d29 100%);
        border-right: 1px solid #ffffff15;
    }
    
    /* Progress Animation */
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.5; }
    }
    
    .stSpinner > div {
        border-color: #667eea !important;
        animation: pulse 1.5s ease-in-out infinite;
    }
    
    /* Code Block Enhancement */
    .stCodeBlock {
        border-radius: 10px;
        border: 1px solid #ffffff15;
        background: #1a1d29 !important;
    }
    
    /* Divider - MINIMAL */
    hr {
        margin: 0.5rem 0 !important;
        border: none;
        height: 1px;
        background: linear-gradient(90deg, transparent, #667eea55, transparent);
    }
    
    /* Reduce all section spacing to MINIMUM */
    .element-container {
        margin-bottom: 0 !important;
        margin-top: 0 !important;
    }
    
    /* Tighter expander spacing */
    div[data-testid="stExpander"] {
        margin: 0.2rem 0 !important;
    }
    
    /* Reduce header margins to MINIMUM */
    h1, h2, h3, h4 {
        margin-top: 0.3rem !important;
        margin-bottom: 0.3rem !important;
        padding: 0 !important;
        line-height: 1.2 !important;
    }
    
    /* Tighten paragraphs - NO margin */
    p {
        margin-bottom: 0 !important;
        margin-top: 0 !important;
    }
    
    /* Reduce form spacing */
    .stForm {
        padding: 0.3rem !important;
        margin: 0 !important;
    }
    
    /* Tighter column spacing */
    [data-testid="column"] {
        padding: 0.2rem !important;
    }
    
    /* Remove spacing from rows */
    .row-widget {
        margin: 0 !important;
        padding: 0 !important;
    }
    
    /* Change top navigation bar color to match heading */
    header[data-testid="stHeader"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
    }
    
    /* Style the toolbar buttons */
    header[data-testid="stHeader"] button {
        color: white !important;
    }
    
    header[data-testid="stHeader"] svg {
        fill: white !important;
        stroke: white !important;
    }
    
    /* Deploy button styling */
    header[data-testid="stHeader"] [data-testid="stToolbar"] {
        color: white !important;
    }
    
    /* Info Boxes - MINIMAL padding and margin */
    .info-box {
        background: linear-gradient(135deg, #667eea15, #764ba215);
        border-left: 4px solid #667eea;
        padding: 0.6rem;
        border-radius: 8px;
        margin: 0.3rem 0;
    }
    
    .success-box {
        background: linear-gradient(135deg, #28a74515, #20c99715);
        border-left: 4px solid #28a745;
        padding: 0.6rem;
        border-radius: 8px;
        margin: 0.3rem 0;
    }
    
    .warning-box {
        background: linear-gradient(135deg, #ffc10715, #ff940015);
        border-left: 4px solid #ffc107;
        padding: 0.6rem;
        border-radius: 8px;
        margin: 0.3rem 0;
    }
    
    .error-box {
        background: linear-gradient(135deg, #dc354515, #e7444815);
        border-left: 4px solid #dc3545;
        padding: 0.6rem;
        border-radius: 8px;
        margin: 0.3rem 0;
    }
    
    /* Section Headers - TIGHT */
    h1, h2, h3 {
        font-weight: 700;
        letter-spacing: -0.5px;
        line-height: 1.2 !important;
    }
    
    h2 {
        background: linear-gradient(135deg, #667eea, #764ba2);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-top: 0.5rem !important;
        margin-bottom: 0.3rem !important;
        font-size: 1.8rem !important;
    }
    
    h3 {
        font-size: 1.3rem !important;
        margin-top: 0.4rem !important;
        margin-bottom: 0.2rem !important;
    }

    /* Remove spacing from sidebar - ULTRA TIGHT */
    [data-testid="stSidebar"] {
        padding-top: 0.3rem !important;
    }
    
    [data-testid="stSidebar"] .block-container {
        padding-top: 0.3rem !important;
        padding-left: 0.4rem !important;
        padding-right: 0.4rem !important;
    }
    
    [data-testid="stSidebar"] .element-container {
        margin-bottom: 0.2rem !important;
    }
    
    [data-testid="stSidebar"] h2 {
        margin-top: 0.3rem !important;
        margin-bottom: 0.3rem !important;
        font-size: 1.2rem !important;
    }
    
    [data-testid="stSidebar"] .info-box,
    [data-testid="stSidebar"] .success-box {
        padding: 0.5rem !important;
        margin: 0.3rem 0 !important;
        font-size: 0.85rem;
    }
    
    /* Tighter metric display */
    [data-testid="stMetric"] {
        padding: 0.2rem !important;
        margin: 0 !important;
    }
    
    [data-testid="stMetricValue"] {
        font-size: 1.8rem !important;
    }
    
    /* Reduce button padding */
    .stButton > button {
        padding: 0.5rem 1.2rem !important;
        margin: 0.2rem 0 !important;
    }
    
    /* Minimal input spacing */
    .stTextInput, .stSelectbox, .stTextArea {
        margin-bottom: 0.3rem !important;
    }
    
    /* Remove extra space from file uploader */
    [data-testid="stFileUploader"] {
        margin: 0.3rem 0 !important;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# INITIALIZE SESSION STATE
# ============================================================================

if 'user_context' not in st.session_state:
    st.session_state.user_context = {}

if 'onboarding_complete' not in st.session_state:
    st.session_state.onboarding_complete = False

if 'analysis_complete' not in st.session_state:
    st.session_state.analysis_complete = False

if 'download_synthetic' not in st.session_state:
    st.session_state.download_synthetic = False

if 'reset_counter' not in st.session_state:
    st.session_state.reset_counter = 0

# ============================================================================
# HEADER
# ============================================================================

st.markdown('<p class="main-header">🏥 MedGuard AI</p>', unsafe_allow_html=True)
st.markdown('<p class="subheader">Multi-Agent Healthcare Data Validator with Privacy-Preserving Intelligence</p>', unsafe_allow_html=True)


# ============================================================================
# SIDEBAR
# ============================================================================

with st.sidebar:
    st.markdown("## 🎯 About MedGuard AI")
    st.markdown("""
    <div class="info-box">
    <strong>🤖 Multi-Agent System:</strong><br>
    • <strong>Medical Advisor</strong> - Clinical context<br>
    • <strong>Fairness Specialist</strong> - Bias mitigation<br>
    • <strong>Deployment Strategist</strong> - Action plans
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="success-box">
    <strong>🔒 HIPAA Protection:</strong><br>
    • Privacy-preserving synthetic data<br>
    • Multi-stage PHI detection<br>
    • Local processing only<br>
    • Zero data transmission
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("## 📁 Demo Datasets")
    demo_option = st.selectbox(
        "Load example:",
        ["Upload Your Own",
         "Demo 1: Heart Disease (Quality Issues)",
         "Demo 2: Diabetes (Gender Bias)",
         "Demo 3: Heart Disease (Indigenous Bias)",
         "Demo 4: Combined Problems"],
        key=f"demo_selector_{st.session_state.get('reset_counter', 0)}"
    )
    
    if st.button("🔄 Reset Analysis", use_container_width=True):
        # Store reset counter before clearing
        reset_count = st.session_state.get('reset_counter', 0) + 1
        
        # Clear ALL session state variables
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        
        # Re-initialize essential variables
        st.session_state.user_context = {}
        st.session_state.onboarding_complete = False
        st.session_state.analysis_complete = False
        st.session_state.download_synthetic = False
        st.session_state.reset_counter = reset_count
        
        st.rerun()
    
    st.markdown("""
    <div style="text-align: center; padding: 0.5rem; opacity: 0.6; margin-top: 1rem;">
    <small>Powered by Llama 3.2 & Ollama</small>
    </div>
    """, unsafe_allow_html=True)

# ============================================================================
# PHASE 1: USER ONBOARDING (Context Gathering)
# ============================================================================

if not st.session_state.onboarding_complete:
    st.markdown("## 📋 Step 1: Tell Us About Your Project")
    st.write("Help us provide context-aware recommendations by answering a few questions:")
    
    with st.form("onboarding_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            project_description = st.text_area(
                "🎯 What are you building?",
                placeholder="e.g., Lung disease prediction model for hospital deployment",
                height=120,
                help="Describe your ML project in 1-2 sentences"
            )
            
            model_type = st.selectbox(
                "🤖 What model will you use?",
                ["Random Forest", "Neural Network", "Logistic Regression",
                 "XGBoost", "Support Vector Machine", "Ensemble", "Other"]
            )
            
            use_case = st.selectbox(
                "🏥 What's your use case?",
                ["Research study", "Clinical deployment",
                 "FDA submission", "Proof of concept", "Academic project"]
            )
        
        with col2:
            timeline = st.selectbox(
                "⏱️ When do you need this deployed?",
                ["<30 days (urgent)", "30-60 days", "60-90 days", "90+ days (flexible)"]
            )
            
            can_collect_data = st.radio(
                "📊 Can you collect additional patient data if needed?",
                ["Yes, we have recruitment capabilities",
                 "Maybe, but it's difficult",
                 "No, we must use existing data only"]
            )
            
            location = st.text_input(
                "📍 Where is this being deployed? (Optional)",
                placeholder="e.g., Boston Medical Center, Boston, MA",
                help="Optional: Helps AI provide location-specific recommendations"
            )
        
        st.write("")  # Spacing
        submitted = st.form_submit_button("✨ Continue to Analysis", use_container_width=True)
        
        if submitted:
            if not project_description:
                st.error("⚠️ Please fill in project description")
            else:
                timeline_days = {
                    "<30 days (urgent)": 30,
                    "30-60 days": 45,
                    "60-90 days": 75,
                    "90+ days (flexible)": 120
                }
                
                st.session_state.user_context = {
                    'project_description': project_description,
                    'model_type': model_type,
                    'use_case': use_case,
                    'timeline_days': timeline_days[timeline],
                    'can_collect_data': can_collect_data,
                    'location': location if location else 'Not specified'
                }
                st.session_state.onboarding_complete = True
                st.rerun()
    
    # ADD THIS - Stop execution here if onboarding not complete
    st.stop()

# ============================================================================
# PHASE 2: DATA UPLOAD & ANALYSIS
# ============================================================================

# This section will ONLY run after onboarding is complete
elif st.session_state.onboarding_complete:
    
    # Show user context summary with enhanced styling
    with st.expander("📋 Your Project Context", expanded=False):
        ctx = st.session_state.user_context
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("🤖 Model", ctx['model_type'])
        with col2:
            st.metric("⏱️ Timeline", f"{ctx['timeline_days']} days")
        with col3:
            st.metric("🏥 Use Case", ctx['use_case'])
        with col4:
            data_status = "✅ Yes" if "Yes" in ctx['can_collect_data'] else "⚠️ Limited" if "Maybe" in ctx['can_collect_data'] else "❌ No"
            st.metric("📊 Data Collection", data_status)
        
        st.write(f"**📍 Location:** {ctx['location']}")
        st.write(f"**🎯 Project:** {ctx['project_description']}")
    
    
    # File upload with better styling
    st.markdown("## 📂 Step 2: Upload Your Dataset")
    
    uploaded_file = None
    df = None
    
   # Handle demo datasets
if demo_option != "Upload Your Own":
    demo_files = {
        "Demo 1: Heart Disease (Quality Issues)": "data/synthetic/dataset1_heart_disease_quality.csv",
        "Demo 2: Diabetes (Gender Bias)": "data/synthetic/dataset2_diabetes_gender_bias.csv",
        "Demo 3: Heart Disease (Indigenous Bias)": "data/synthetic/dataset3_heart_disease_indigenous.csv",
        "Demo 4: Combined Problems": "data/synthetic/dataset4_diabetes_combined.csv"
    }
    
    demo_path = demo_files[demo_option]
    
    if os.path.exists(demo_path):
        df = pd.read_csv(demo_path)
        
        # PHI SCAN
        scanner = PHIScanner()
        is_safe, violations = scanner.scan_dataset(df)
        
        if not is_safe:
            st.markdown("""
            <div class="error-box">
            <h3>🚨 PHI DETECTED - Cannot Process</h3>
            <strong>Violations found:</strong>
            </div>
            """, unsafe_allow_html=True)
            for v in violations:
                st.write(f"  • {v}")
            st.warning("Please de-identify your dataset and re-upload.")
            st.info("**Remove:** patient names, SSNs, phone numbers, email addresses, dates of birth, addresses")
            st.stop()
        
        # CONSOLIDATED STATUS BOX
        st.markdown(f"""
        <div>
        <strong>✅ Loaded:</strong> {demo_option}<br>
        📊 <strong>Records:</strong> {len(df):,} | <strong>Features:</strong> {len(df.columns)}<br>
        ⚠️ <strong>DEMO MODE:</strong> Using synthetic data for demonstration<br>
        ✅ <strong>PHI Scan Passed</strong> - Dataset is properly de-identified<br>
        🔒 <strong>Privacy Firewall Active</strong>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.error(f"Demo file not found: {demo_path}")
        st.stop()
else:
    uploaded_file = st.file_uploader(
        "Upload your healthcare dataset (CSV)",
        type=['csv'],
        help="Must be de-identified. Our tool will validate.",
        label_visibility="collapsed"
    )
    
    if uploaded_file is None:
        st.markdown("""
        <div class="info-box">
        👆 <strong>Upload a CSV file</strong> or select a demo dataset from the sidebar to begin analysis
        </div>
        """, unsafe_allow_html=True)
        st.stop()
    
    df = pd.read_csv(uploaded_file)
    
    # PHI SCAN
    with st.spinner("🔍 Scanning for Protected Health Information..."):
        scanner = PHIScanner()
        is_safe, violations = scanner.scan_dataset(df)
    
    if not is_safe:
        st.markdown("""
        <div class="error-box">
        <h3>🚨 PHI DETECTED - Cannot Process</h3>
        <strong>Violations found:</strong>
        </div>
        """, unsafe_allow_html=True)
        for v in violations:
            st.write(f"  • {v}")
        st.warning("Please de-identify your dataset and re-upload.")
        st.info("**Remove:** patient names, SSNs, phone numbers, email addresses, dates of birth, addresses")
        st.stop()
    
    # CONSOLIDATED STATUS BOX FOR UPLOADED FILE
    st.markdown(f"""
    <div>
    ✅ <strong>File loaded successfully!</strong><br>
    📊 <strong>Records:</strong> {len(df):,} | <strong>Features:</strong> {len(df.columns)}<br>
    ✅ <strong>PHI Scan Passed</strong> - Dataset is properly de-identified<br>
    🔒 <strong>Privacy Firewall Active</strong> 
    </div>
    """, unsafe_allow_html=True)
    
    # ========================================================================
    # GENERATE SYNTHETIC TWIN (PRIVACY FIREWALL)
    # ========================================================================


# Add synthetic dataset generation controls
col_generate1, col_generate2 = st.columns([3, 1])
with col_generate1:
    synthetic_size = st.slider(
        "Synthetic dataset size",
        min_value=100,
        max_value=min(5000, len(df) * 3),
        value=len(df),
        step=100,
        help="Number of synthetic records to generate"
    )

with col_generate2:
    st.write("")
    if st.button("📥 Download Synthetic", use_container_width=True):
        st.session_state.download_synthetic = True

# Cache synthetic dataset generation to avoid regenerating on every rerun
# Create a hash based on original df and synthetic_size
try:
    # Try to use values hash for numeric data (faster)
    df_hash = hash(str(df.values.tobytes()) + str(df.columns.tolist()) + str(len(df)) + str(df.shape))
except (AttributeError, TypeError):
    # Fallback: use string representation (works with all data types)
    df_hash = hash(str(df.to_string()) + str(df.columns.tolist()) + str(len(df)) + str(df.shape))
synthetic_cache_key = f"synthetic_df_{df_hash}_{synthetic_size}"

# Check if we need to regenerate (only if df changed or size changed)
if (synthetic_cache_key not in st.session_state or 
    'synthetic_df' not in st.session_state or
    'synthetic_validation' not in st.session_state):
    
    with st.spinner("⚙️ Generating synthetic twin..."):
        generator = SyntheticDataGenerator()
        generator.fit(df)
        synthetic_df = generator.generate(n_samples=synthetic_size)
        validation = generator.validate_privacy(df, synthetic_df)
    
    # Cache the synthetic dataset and validation
    st.session_state.synthetic_df = synthetic_df
    st.session_state.synthetic_validation = validation
    st.session_state.synthetic_cache_key = synthetic_cache_key
else:
    # Use cached synthetic dataset (instant - no regeneration!)
    synthetic_df = st.session_state.synthetic_df
    validation = st.session_state.synthetic_validation

# Continue with rest of the code...
if validation['privacy_safe']:
        similarity = validation['statistical_similarity']
        st.write(f"✅ Synthetic twin generated | Statistical similarity: **{similarity:.1%}** ")
        
        # Download synthetic dataset if requested
        if st.session_state.get('download_synthetic', False):
            csv_data = synthetic_df.to_csv(index=False)
            st.download_button(
                label="💾 Download Synthetic CSV",
                data=csv_data,
                file_name=f"synthetic_dataset_{synthetic_size}_records.csv",
                mime="text/csv",
                use_container_width=True
            )
            st.success("✅ Click the button above to download your synthetic dataset!")
            st.session_state.download_synthetic = False
        
        # Show sample comparison
        with st.expander("📊 Compare Original vs Synthetic Data", expanded=False):
            comp_col1, comp_col2 = st.columns(2)
            
            with comp_col1:
                st.markdown("**Original Data (Sample)**")
                st.dataframe(df.head(5), use_container_width=True)
                
            with comp_col2:
                st.markdown("**Synthetic Data (Sample)**")
                st.dataframe(synthetic_df.head(5), use_container_width=True)
            
            # Statistical comparison
            st.markdown("**Statistical Comparison**")
            
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                comparison_data = []
                for col in numeric_cols[:5]:
                    comparison_data.append({
                        'Feature': col,
                        'Original Mean': f"{df[col].mean():.2f}",
                        'Synthetic Mean': f"{synthetic_df[col].mean():.2f}",
                        'Original Std': f"{df[col].std():.2f}",
                        'Synthetic Std': f"{synthetic_df[col].std():.2f}",
                        'Similarity': f"{(1 - abs(df[col].mean() - synthetic_df[col].mean()) / df[col].mean() * 100):.1f}%" if df[col].mean() != 0 else "N/A"
                    })
                
                comparison_df = pd.DataFrame(comparison_data)
                st.dataframe(comparison_df, use_container_width=True)
else:
    st.warning("⚠️ Privacy validation concern - proceeding with caution")

# ========================================================================
# ANALYSIS WITH AI AGENTS
# ========================================================================

#Sst.divider()
st.markdown("## 🤖 Step 5: Multi-Agent Analysis")

# Cache analysis results to avoid rerunning on every widget interaction
# Create a hash of the dataset to detect if it changed
# Use a more robust hash that works with mixed data types
try:
    dataset_hash = hash(str(synthetic_df.values.tobytes()) + str(synthetic_df.columns.tolist()) + str(len(synthetic_df)))
except (AttributeError, TypeError):
    # Fallback for non-numeric data: use string representation
    dataset_hash = hash(str(synthetic_df.to_string()) + str(synthetic_df.columns.tolist()) + str(len(synthetic_df)))

# Check if we need to run analysis (only if dataset changed or results not cached)
if ('analysis_hash' not in st.session_state or 
    st.session_state.analysis_hash != dataset_hash or
    'quality_issues' not in st.session_state or
    'bias_issues' not in st.session_state):
    
    # Run Python analysis (only once per dataset)
    with st.spinner("🔬 Running comprehensive statistical analysis..."):
        analyzer = DataQualityAnalyzer(synthetic_df)
        quality_issues = analyzer.run_full_analysis()
        quality_summary = analyzer.get_summary()
        
        # Detect target column
        possible_targets = [c for c in synthetic_df.columns 
                          if any(t in c.lower() for t in ['disease', 'diabetes', 'outcome', 'target', 'recovered'])]
        target_col = possible_targets[0] if possible_targets else synthetic_df.columns[-1]
        
        bias_detector = BiasDetector(synthetic_df, target_col=target_col)
        bias_issues = bias_detector.run_full_analysis()
        bias_summary = bias_detector.get_summary()
    
    # Cache the results
    st.session_state.quality_issues = quality_issues
    st.session_state.quality_summary = quality_summary
    st.session_state.bias_issues = bias_issues
    st.session_state.bias_summary = bias_summary
    st.session_state.analysis_hash = dataset_hash
else:
    # Use cached results (instant - no computation!)
    quality_issues = st.session_state.quality_issues
    quality_summary = st.session_state.quality_summary
    bias_issues = st.session_state.bias_issues
    bias_summary = st.session_state.bias_summary

# ========================================================================
# PRE-COMPUTE RECOMMENDATIONS FOR "IMPLEMENT CHANGES" TAB
# ========================================================================
# Cache recommendations immediately after analysis so tab5 loads instantly
# Only recompute if quality_issues or bias_issues changed

def _collect_all_recommendations():
    """Collect all recommendations including quality, bias normalization, and bias mitigation"""
    quality_issues = st.session_state.get('quality_issues', {})
    bias_issues = st.session_state.get('bias_issues', {})
    
    all_recommendations = []
    
    # Collect missing value recommendations
    if quality_issues.get('missing_values'):
        for col, info in quality_issues['missing_values'].items():
            for rec in info.get('recommendations', []):
                if 'code' in rec and rec.get('priority') not in ['❌ NOT RECOMMENDED', '❌ CRITICAL']:
                    all_recommendations.append({
                        'type': 'missing_value',
                        'column': col,
                        'method': rec['method'],
                        'priority': rec['priority'],
                        'reason': rec['reason'],
                        'code': rec['code'],
                        'value': rec.get('value'),
                        'impact': rec.get('impact', ''),
                        'key': f"missing_{col}_{rec['method'].replace(' ', '_').lower()}"
                    })
    
    # Collect duplicate recommendations
    if quality_issues.get('duplicates'):
        dup = quality_issues['duplicates']
        if dup.get('recommendation'):
            rec = dup['recommendation']
            all_recommendations.append({
                'type': 'duplicate',
                'column': None,
                'method': rec['method'],
                'priority': rec['priority'],
                'reason': rec['reason'],
                'code': rec['code'],
                'value': None,
                'impact': rec.get('impact', ''),
                'key': 'duplicate_remove'
            })
    
    # Collect outlier recommendations
    if quality_issues.get('outliers'):
        for col, info in quality_issues['outliers'].items():
            for rec in info.get('recommendations', []):
                if 'code' in rec and rec.get('priority') not in ['❌ NOT RECOMMENDED']:
                    all_recommendations.append({
                        'type': 'outlier',
                        'column': col,
                        'method': rec['method'],
                        'priority': rec['priority'],
                        'reason': rec['reason'],
                        'code': rec['code'],
                        'value': None,
                        'impact': rec.get('impact', ''),
                        'key': f"outlier_{col}_{rec['method'].replace(' ', '_').lower()}"
                    })
    
    # Collect bias normalization recommendations
    if bias_issues:
        for col, info in bias_issues.items():
            if 'sex' in col.lower() or 'gender' in col.lower():
                dist_keys = list(info.get('distribution', {}).keys())
                if len(dist_keys) > 2 or any(k not in ['Male', 'Female'] for k in dist_keys):
                    all_recommendations.append({
                        'type': 'bias_normalization',
                        'column': col,
                        'method': 'Normalize Sex/Gender Values',
                        'priority': '⭐ RECOMMENDED',
                        'reason': f'Normalize mixed encodings (Male/Female/M/F/1/0) to standard Male/Female format',
                        'code': f"# Normalize {col} values",
                        'value': None,
                        'impact': 'Ensures consistent demographic representation',
                        'key': f'bias_normalize_{col}'
                    })
    
    # Collect bias mitigation recommendations from Fairness Specialist (SMOTE, etc.)
    if 'fairness_recommendations' in st.session_state and st.session_state.fairness_recommendations:
        for col, fairness_rec in st.session_state.fairness_recommendations.items():
            # Only add if there's actual bias (imbalance detected)
            if fairness_rec.severity in ['critical', 'high', 'medium']:
                tech_fix = fairness_rec.immediate_technical_fix
                all_recommendations.append({
                    'type': 'bias_mitigation',
                    'column': col,
                    'method': tech_fix.method,
                    'priority': '⭐ RECOMMENDED' if fairness_rec.severity in ['critical', 'high'] else '⚡ OPTIONAL',
                    'reason': f'Balance {col} distribution using {tech_fix.method}. {fairness_rec.immediate_technical_fix.expected_improvement}',
                    'code': tech_fix.python_code,
                    'value': None,
                    'impact': tech_fix.expected_improvement,
                    'key': f'bias_mitigation_{col}_{tech_fix.method.lower()}',
                    'fairness_rec': fairness_rec,  # Store full recommendation for implementation
                    'distribution': fairness_rec.current_distribution,
                    'target_distribution': fairness_rec.target_distribution
                })
    
    # Also add basic bias mitigation recommendations directly from bias_issues
    # (in case Fairness Specialist hasn't run yet or failed)
    if bias_issues:
        for col, info in bias_issues.items():
            if info.get('bias_detected', False):
                # Check if we already have a recommendation for this column
                has_recommendation = any(
                    rec.get('column') == col and rec.get('type') == 'bias_mitigation'
                    for rec in all_recommendations
                )
                if not has_recommendation:
                    # Add basic SMOTE recommendation
                    dist_values = list(info.get('distribution', {}).values())
                    if len(dist_values) >= 2:
                        gap = abs(max(dist_values) - min(dist_values))
                        if gap > 5:  # Significant imbalance
                            all_recommendations.append({
                                'type': 'bias_mitigation',
                                'column': col,
                                'method': 'SMOTE',
                                'priority': '⭐ RECOMMENDED',
                                'reason': f'Balance {col} distribution (current gap: {gap:.1f}%)',
                                'code': 'from imblearn.over_sampling import SMOTE\nsmote = SMOTE(random_state=42)\nX_balanced, y_balanced = smote.fit_resample(X, y)',
                                'value': None,
                                'impact': f'Reduce distribution gap from {gap:.1f}% to <5%',
                                'key': f'bias_mitigation_{col}_smote'
                            })
    
    # Cache the recommendations and grouped structure
    st.session_state.cached_recommendations = all_recommendations
    
    # Pre-group recommendations by type for faster rendering
    recommendations_by_type = {}
    for rec in all_recommendations:
        rec_type = rec['type']
        if rec_type not in recommendations_by_type:
            recommendations_by_type[rec_type] = []
        recommendations_by_type[rec_type].append(rec)
    st.session_state.cached_recommendations_grouped = recommendations_by_type

# Collect recommendations initially
current_issues_hash = str(hash(str(quality_issues) + str(bias_issues)))
if 'issues_hash' not in st.session_state or st.session_state.issues_hash != current_issues_hash:
    _collect_all_recommendations()
    st.session_state.issues_hash = current_issues_hash
else:
    # Even if hash matches, re-collect if fairness_recommendations were added
    # (they might have been added after initial collection)
    fairness_hash = str(hash(str(st.session_state.get('fairness_recommendations', {}))))
    if 'fairness_hash' not in st.session_state or st.session_state.fairness_hash != fairness_hash:
        _collect_all_recommendations()
        st.session_state.fairness_hash = fairness_hash

# Enhanced summary metrics
col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.metric("📊 Total Records", f"{quality_summary['total_records']:,}")
with col2:
    missing_count = quality_summary['missing_value_columns']
    st.metric("⚠️ Missing Values", missing_count, delta=f"-{missing_count}" if missing_count > 0 else None, delta_color="inverse")
with col3:
    dup_count = quality_summary.get('duplicate_records', 0)
    st.metric("🔄 Duplicates", dup_count, delta=f"-{dup_count}" if dup_count > 0 else None, delta_color="inverse")
with col4:
    bias_count = bias_summary['attributes_with_bias']
    st.metric("⚖️ Bias Detected", bias_count, delta=f"-{bias_count}" if bias_count > 0 else None, delta_color="inverse")
with col5:
    total_issues = missing_count + dup_count + bias_count
    if total_issues == 0:
        st.metric("🎯 Status", "🟢 Clean")
    elif total_issues <= 3:
        st.metric("🎯 Status", "🟡 Minor")
    else:
        st.metric("🎯 Status", "🔴 Critical")

# ========================================================================
# TABS
# ========================================================================

st.divider()

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🔍 Data Quality + AI Insights",
    "⚖️ Bias Analysis + Fairness Plan",
    "📅 Deployment Roadmap",
    "💬 Ask AI Questions",
    "🛠️ Implement Changes"
])

# ========================================================================
# TAB 1: DATA QUALITY (continued in next artifact section...)
# ========================================================================

with tab1:
        st.markdown("# 🔍 Data Quality Analysis")
        st.caption("Powered by Statistical Analyzer + Medical Advisor AI")
        
        # Initialize Medical Advisor
        if 'medical_advisor' not in st.session_state:
            with st.spinner("🤖 Loading Medical Advisor AI..."):
                try:
                    st.session_state.medical_advisor = MedicalAdvisorV2()
                except Exception as e:
                    st.warning(f"Medical Advisor unavailable: {e}")
                    st.session_state.medical_advisor = None
        
        medical_advisor = st.session_state.medical_advisor
        
        # Missing Values
        if quality_issues['missing_values']:
            st.markdown("### ⚠️ Missing Values Detected")
            
            for col, info in quality_issues['missing_values'].items():
                with st.expander(f"❌ **{col}**: {info['count']} missing ({info['percentage']}%)", expanded=False):
                    
                    col_a, col_b = st.columns([1, 1])
                    
                    with col_a:
                        st.markdown("#### 📊 Statistics")
                        st.markdown(f"""
                        <div class="warning-box">
                        • <strong>Missing:</strong> {info['count']} values ({info['percentage']}%)<br>
                        • <strong>Affected rows:</strong> {info['total_affected_rows']}
                        </div>
                        """, unsafe_allow_html=True)
                        
                        st.markdown("#### 🔧 Verified Python Recommendations")
                        for i, rec in enumerate(info['recommendations'][:3], 1):
                            st.markdown(f"**{i}. {rec['method']}** {rec['priority']}")
                            st.write(f"💡 {rec['reason']}")
                            st.code(rec['code'], language='python')
                            if 'impact' in rec:
                                st.caption(f"✨ Impact: {rec['impact']}")
                            if i < 3:
                                st.write("")
                    
                    with col_b:
                        st.markdown("#### 🤖 AI Medical Context")
                        
                        if medical_advisor:
                            with st.spinner("🔬 Medical Advisor analyzing..."):
                                if col.lower() in ['cholesterol', 'glucose', 'blood_pressure', 'bmi', 'lung_capacity', 'lung capacity']:
                                    st.markdown(f"""
                                    <div class="info-box">
                                    <strong>Clinical Significance:</strong> Missing {col} data
                                    </div>
                                    """, unsafe_allow_html=True)
                                    
                                    if 'cholesterol' in col.lower():
                                        st.markdown("""
                                        <div class="success-box">
                                        <strong>Medical Advisor:</strong><br>
                                        Cholesterol is a primary cardiovascular risk biomarker. Median imputation 
                                        falls in 'borderline-high' range, clinically appropriate for cardiac cohorts.
                                        High outliers (>400) should be KEPT - they represent highest-risk patients.
                                        </div>
                                        """, unsafe_allow_html=True)
                                    elif 'lung' in col.lower():
                                        st.markdown("""
                                        <div class="success-box">
                                        <strong>Medical Advisor:</strong><br>
                                        Lung capacity is critical for respiratory function assessment. Missing values
                                        may indicate patients too ill to perform pulmonary function tests. Consider
                                        if missingness correlates with disease severity (MNAR pattern).
                                        </div>
                                        """, unsafe_allow_html=True)
                                    elif 'glucose' in col.lower():
                                        st.markdown("""
                                        <div class="success-box">
                                        <strong>Medical Advisor:</strong><br>
                                        Glucose is essential for diabetes screening. Values >126 mg/dL indicate
                                        diabetes. High outliers are clinically meaningful - represent uncontrolled
                                        diabetes cases critical for prediction.
                                        </div>
                                        """, unsafe_allow_html=True)
                                else:
                                    st.info("Standard statistical imputation recommended for this feature.")
                        else:
                            st.info("Medical Advisor unavailable - showing statistical recommendations only")
        else:
            st.markdown("""
            <div class="success-box">
            ✅ <strong>No missing values detected!</strong> Your dataset is complete.
            </div>
            """, unsafe_allow_html=True)
        
        # Duplicates
        st.divider()
        st.markdown("### 🔄 Duplicate Records")
        
        if quality_issues['duplicates']:
            dup = quality_issues['duplicates']
            st.markdown(f"""
            <div class="warning-box">
            ⚠️ Found <strong>{dup['count']} duplicate records</strong>
            </div>
            """, unsafe_allow_html=True)
            
            rec = dup['recommendation']
            st.write(f"**{rec['priority']} {rec['method']}**")
            st.write(rec['reason'])
            st.code(rec['code'], language='python')
            st.info(f"💡 {rec['impact']}")
        else:
            st.markdown("""
            <div class="success-box">
            ✅ <strong>No duplicates detected!</strong> Each record is unique.
            </div>
            """, unsafe_allow_html=True)
        
        # Outliers
        st.divider()
        st.markdown("### 📈 Outlier Detection")
        
        if quality_issues['outliers']:
            for col, info in quality_issues['outliers'].items():
                with st.expander(f"⚡ **{col}**: {info['count']} outliers detected"):
                    st.write(f"**📊 Sample values:** {info['values_sample']}")
                    st.write(f"**📏 Expected range:** {info['bounds']['lower']:.1f} - {info['bounds']['upper']:.1f}")
                    
                    st.markdown("#### 🔧 Recommendations:")
                    for rec in info['recommendations']:
                        st.markdown(f"**{rec['method']}** {rec['priority']}")
                        st.write(rec['reason'])
                        if 'code' in rec:
                            st.code(rec['code'], language='python')
        else:
            st.markdown("""
            <div class="success-box">
            ✅ <strong>No significant outliers!</strong> Data distribution looks healthy.
            </div>
            """, unsafe_allow_html=True)

# ========================================================================
# TAB 2: BIAS ANALYSIS + FAIRNESS SPECIALIST
# ========================================================================

with tab2:
        st.markdown("# ⚖️ Bias & Fairness Analysis")
        st.caption("Powered by Fairness Specialist AI + Statistical Analysis")
        
        # Initialize Fairness Specialist
        if 'fairness_specialist' not in st.session_state:
            with st.spinner("🤖 Loading Fairness Specialist AI..."):
                try:
                    st.session_state.fairness_specialist = FairnessSpecialist()
                except Exception as e:
                    st.warning(f"Fairness Specialist unavailable: {e}")
                    st.session_state.fairness_specialist = None
        
        fairness_specialist = st.session_state.fairness_specialist
        
        if bias_issues:
            for col, info in bias_issues.items():
                st.markdown(f"### Analysis: {col}")
                
                # Create distribution visualization
                dist_df = pd.DataFrame({
                    'Group': list(info['distribution'].keys()),
                    'Percentage': list(info['distribution'].values())
                })
                
                # Create bar chart
                fig = go.Figure()
                
                colors = []
                for pct in dist_df['Percentage']:
                    deviation = abs(pct - 50)
                    if deviation < 5:
                        colors.append('#28a745')
                    elif deviation < 15:
                        colors.append('#ffc107')
                    else:
                        colors.append('#dc3545')
                
                fig.add_trace(go.Bar(
                    x=dist_df['Group'],
                    y=dist_df['Percentage'],
                    text=[f"<b>{p:.1f}%</b>" for p in dist_df['Percentage']],
                    textposition='outside',
                    textfont=dict(size=16, color='white'),
                    marker=dict(color=colors, line=dict(color='white', width=2)),
                    hovertemplate='<b>%{x}</b><br>%{y:.2f}%<extra></extra>'
                ))
                
                fig.add_hline(
                    y=50,
                    line_dash="dash",
                    line_width=3,
                    line_color="white",
                    annotation_text="◄ Target: 50% (Balanced)",
                    annotation_position="right",
                    annotation_font_size=12
                )
                
                fig.update_layout(
                    title=dict(text=f'<b>{col} Distribution</b>', font=dict(size=22, color='white')),
                    xaxis_title='Group',
                    yaxis_title='Percentage (%)',
                    yaxis=dict(range=[0, 100], dtick=10, gridcolor='#444'),
                    xaxis=dict(gridcolor='#444'),
                    showlegend=False,
                    height=500,
                    template='plotly_dark',
                    plot_bgcolor='#0e1117',
                    paper_bgcolor='#0e1117'
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Show interpretation based on bias detection
                max_pct = max(dist_df['Percentage'])
                min_pct = min(dist_df['Percentage'])
                gap = abs(max_pct - min_pct)
                
                # Bias recommendations
                if info['bias_detected']:
                    # Show gap interpretation only when bias is detected
                    if gap < 15:
                        st.warning(f"⚠️ **Minor imbalance detected.** Distribution gap: {gap:.1f}% (FDA threshold: <5%)")
                    else:
                        st.error(f"🚨 **Significant imbalance!** Distribution gap: {gap:.1f}% (FDA threshold: <5%)")
                    st.markdown("""
                    <div class="error-box">
                    ⚠️ <strong>BIAS DETECTED - Detailed Issues:</strong>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    for issue in info['issues']:
                        st.write(f"  • {issue}")
                    
                    st.markdown("---")
                    st.markdown("### 🤖 AI-Generated Fairness Plan")
                    
                    if fairness_specialist:
                        with st.spinner("Fairness Specialist generating recommendations..."):
                            bias_data = {
                                'type': f'{col}_imbalance',
                                'distribution': info['distribution'],
                                'minority_group': info['issues'][0].split(':')[0].strip() if info['issues'] else 'underrepresented'
                            }
                            
                            dist_values = list(info['distribution'].values())
                            majority_pct = max(dist_values)
                            minority_pct = min(dist_values)
                            
                            total = len(synthetic_df)
                            majority_count = int(total * majority_pct / 100)
                            minority_count = int(total * minority_pct / 100)
                            samples_needed = majority_count - minority_count
                            
                            stats = {
                                'total_samples': total,
                                'samples_needed': samples_needed,
                                'imbalance_pct': abs(majority_pct - 50)
                            }
                            
                            try:
                                fairness_rec = fairness_specialist.analyze_bias(bias_data, st.session_state.user_context, stats)
                                
                                # Store fairness recommendations in session state for Tab 5 (Implement Changes)
                                if 'fairness_recommendations' not in st.session_state:
                                    st.session_state.fairness_recommendations = {}
                                st.session_state.fairness_recommendations[col] = fairness_rec
                                
                                # Re-collect recommendations to include this new bias mitigation recommendation
                                # This ensures Tab 5 shows bias mitigation options even if collected before Tab 2 ran
                                _collect_all_recommendations()
                                
                                st.success("✅ Exact Fairness Recommendations Generated by AI")
                                
                                # Predicted Harm
                                st.markdown("#### 🚨 Predicted Harm if Not Fixed:")
                                harm_col1, harm_col2, harm_col3 = st.columns(3)
                                
                                with harm_col1:
                                    st.metric("Performance Gap", f"{fairness_rec.predicted_harm.performance_gap_percentage:.1f}%",
                                             delta=f"-{fairness_rec.predicted_harm.performance_gap_percentage:.1f}%", delta_color="inverse")
                                
                                with harm_col2:
                                    st.metric("Patients Affected", f"{fairness_rec.predicted_harm.patients_affected_per_100}/100")
                                
                                with harm_col3:
                                    severity_color = "🔴" if fairness_rec.severity == "critical" else "🟡"
                                    st.metric("Severity", f"{severity_color} {fairness_rec.severity.upper()}")
                                
                                st.error(f"**Clinical Impact:** {fairness_rec.predicted_harm.clinical_consequence}")
                                
                                # Exact Requirements
                                st.markdown("---")
                                st.markdown("#### 🔢 Exact Requirements for Fairness:")
                                for group, count in fairness_rec.exact_samples_needed.items():
                                    st.write(f"  • **{group}:** Need **{count:,} additional patients**")
                                
                                # Two Options Side-by-Side
                                st.markdown("---")
                                st.markdown("#### 🛠️ Remediation Options:")
                                
                                option_col1, option_col2 = st.columns(2)
                                
                                with option_col1:
                                    st.markdown("**🏥 Option 1: Real Data Collection**")
                                    st.write(f"**⏱️ Timeline:** {fairness_rec.recruitment_plan.timeline_months} months")
                                    
                                    if fairness_rec.recruitment_plan.estimated_cost_usd:
                                        st.write(f"**💰 Cost:** ${fairness_rec.recruitment_plan.estimated_cost_usd:,}")
                                    
                                    st.write("\n**🏥 Target Facilities:**")
                                    for i, facility in enumerate(fairness_rec.recruitment_plan.target_facilities, 1):
                                        st.write(f"  {i}. {facility}")
                                    
                                    user_timeline = st.session_state.user_context['timeline_days']
                                    recruitment_days = fairness_rec.recruitment_plan.timeline_months * 30
                                    
                                    if not fairness_rec.fits_user_timeline:
                                        st.error(f"❌ **Timeline Mismatch:** Your timeline ({user_timeline} days) is shorter than needed ({recruitment_days} days)")
                                    else:
                                        st.success(f"✅ Fits your {user_timeline}-day timeline")
                                
                                with option_col2:
                                    st.markdown("**⚡ Option 2: Immediate Technical Fix**")
                                    st.write(f"**🔧 Method:** {fairness_rec.immediate_technical_fix.method}")
                                    st.write(f"**📈 Expected:** {fairness_rec.immediate_technical_fix.expected_improvement}")
                                    
                                    st.code(fairness_rec.immediate_technical_fix.python_code, language='python')
                                    st.warning(f"⚠️ **Limitation:** {fairness_rec.immediate_technical_fix.limitations}")
                                
                                # Priority Recommendation
                                st.markdown("---")
                                st.info(f"💡 **AI Strategic Recommendation:** {fairness_rec.recommendation_priority}")
                                
                            except Exception as e:
                                st.error(f"⚠️ AI fairness analysis failed: {e}")
                                st.code(f"""from imblearn.over_sampling import SMOTE
smote = SMOTE(random_state=42)
X_balanced, y_balanced = smote.fit_resample(X, y)""", language='python')
                    else:
                        st.info("Fairness Specialist unavailable - showing statistical metrics only")
                else:
                    # Show success message when no bias detected
                    max_pct = max(dist_df['Percentage'])
                    min_pct = min(dist_df['Percentage'])
                    gap = abs(max_pct - min_pct)
                    st.success(f"✅ **Well balanced!** Distribution gap: {gap:.1f}% (FDA threshold: <5%)")
        else:
            st.markdown("""
            <div class="info-box">
            ℹ️ <strong>No demographic columns detected in dataset</strong><br>
            Looking for columns containing: 'sex', 'gender', 'race', 'ethnicity', 'age'
            </div>
            """, unsafe_allow_html=True)

# ========================================================================
# TAB 3: DEPLOYMENT ROADMAP
# ========================================================================

with tab3:
        st.markdown("# 📅 Deployment Roadmap")
        st.caption("Powered by Deployment Strategist AI")
        
        # Initialize strategist
        if 'deployment_strategist' not in st.session_state:
            with st.spinner("🤖 Loading Deployment Strategist AI..."):
                try:
                    st.session_state.deployment_strategist = DeploymentStrategist()
                except Exception as e:
                    st.warning(f"Deployment Strategist unavailable: {e}")
                    st.session_state.deployment_strategist = None
        
        strategist = st.session_state.deployment_strategist
        
        if strategist:
            with st.spinner("🤖 AI generating week-by-week implementation plan..."):
                quality_summary_dict = {
                    'missing_values': len(quality_issues['missing_values']),
                    'duplicates': quality_issues['duplicates'].get('count', 0) if quality_issues['duplicates'] else 0,
                    'outliers': len(quality_issues['outliers'])
                }
                
                bias_summary_dict = {
                    'biased_attributes': bias_summary['attributes_with_bias'],
                    'status': bias_summary['status']
                }
                
                fairness_rec_dict = {
                    'exact_samples_needed': {},
                    'timeline_months': 6,
                    'immediate_method': 'data cleaning'
                }
                
                try:
                    plan = strategist.create_plan(quality_summary_dict, bias_summary_dict, fairness_rec_dict, st.session_state.user_context)
                    
                    
                    # Strategy overview
                    st.markdown("### Recommended Strategy")
                    st.markdown(f"""
                    <div class="info-box">
                    <strong>{plan.recommended_strategy}</strong><br><br>
                    <strong>Rationale:</strong> {plan.why_this_strategy}
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Weekly breakdown
                    st.markdown("---")
                    st.markdown("### 📅 Week-by-Week Implementation:")
                    
                    for week in plan.weekly_plan:
                        with st.expander(f"📆 Week {week.week_number}: {week.days_range}", expanded=False):
                            st.markdown("<strong>Tasks:</strong>", unsafe_allow_html=True)

                            for task in week.tasks:
                                st.write(f" {task}")
                            
                            st.markdown("<br><strong>Deliverables:</strong>", unsafe_allow_html=True)
                            for deliverable in week.deliverables:
                                st.write(f"  {deliverable}")
                            
                            if week.estimated_hours:
                                st.caption(f"⏱️ Time estimate: {week.estimated_hours} hours")
                            st.write("")
                    
                    # Critical info
                    st.markdown("---")
                    
                    crit_col1, crit_col2 = st.columns(2)
                    
                    with crit_col1:
                        st.markdown("### 🎯 Critical Path Items:")
                        st.markdown("""
                        <div class="error-box">
                        """, unsafe_allow_html=True)
                        for item in plan.critical_path_items:
                            st.write(f"  🔴 {item}")
                        st.markdown("</div>", unsafe_allow_html=True)
                        
                        st.markdown("\n### ⚠️ Risk Factors:")
                        st.markdown("""
                        <div class="warning-box">
                        """, unsafe_allow_html=True)
                        for risk in plan.risk_factors:
                            st.write(f"  ⚠️ {risk}")
                        st.markdown("</div>", unsafe_allow_html=True)
                    
                    with crit_col2:
                        st.markdown("### 📊 Success Metrics:")
                        st.markdown("""
                        <div class="success-box">
                        """, unsafe_allow_html=True)
                        for metric in plan.success_metrics:
                            st.write(f"  ✅ {metric}")
                        st.markdown("</div>", unsafe_allow_html=True)
                        
                        if plan.alternative_paths:
                            st.markdown("\n### 🔀 Alternative Paths:")
                            for alt in plan.alternative_paths:
                                with st.expander(f"If: {alt.scenario}"):
                                    st.write(f"**When to use:** {alt.when_to_pivot}")
                                    st.write(f"**Modified plan:** {alt.modified_timeline}")
                    
                except Exception as e:
                    st.error(f"⚠️ Plan generation failed: {e}")
                    st.markdown("""
                    <div class="info-box">
                    <strong>📋 Recommended Actions:</strong><br>
                    1. <strong>Week 1:</strong> Address data quality issues (missing values, duplicates)<br>
                    2. <strong>Week 2-3:</strong> Apply bias mitigation if needed<br>
                    3. <strong>Week 4+:</strong> Model training and validation
                    </div>
                    """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="info-box">
            <strong>📋 General Recommendations:</strong><br><br>
            <strong>Phase 1 (Weeks 1-2):</strong> Data Quality<br>
            • Fix missing values using recommended imputation methods<br>
            • Remove duplicate records<br>
            • Handle outliers based on clinical context<br><br>
            <strong>Phase 2 (Weeks 3-4):</strong> Bias Mitigation<br>
            • Apply demographic balancing techniques<br>
            • Validate fairness metrics<br><br>
            <strong>Phase 3 (Weeks 5+):</strong> Model Development<br>
            • Train and validate model<br>
            • Test for bias and performance gaps<br>
            • Prepare deployment documentation
            </div>
            """, unsafe_allow_html=True)

# ========================================================================
# TAB 4: INTERACTIVE Q&A
# ========================================================================

with tab4:
        st.markdown("# 💬 Ask the AI Agents")
        st.caption("Get clarification on any recommendation")
        
        st.markdown("""
        <div class="info-box">
        <strong>💡 Example questions:</strong><br>
        • What if I can only recruit 200 patients instead of 552?<br>
        • Why is median better than mean for my data?<br>
        • How does SMOTE work?<br>
        • What are the risks of deploying with synthetic data?<br>
        • Should I remove or keep high cholesterol values?
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        user_question = st.text_area(
            "🤔 Your question:",
            placeholder="Ask anything about the analysis or recommendations...",
            height=100
        )
        
        if st.button("🚀 Ask AI", use_container_width=True):
            if user_question:
                with st.spinner("🤖 AI agents thinking..."):
                    try:
                        # Build context from analysis
                        context_summary = f"""
User's project:
- Model: {st.session_state.user_context.get('model_type', 'Unknown')}
- Timeline: {st.session_state.user_context.get('timeline_days', 'Unknown')} days
- Use case: {st.session_state.user_context.get('use_case', 'Unknown')}
- Location: {st.session_state.user_context.get('location', 'Unknown')}

Dataset analysis:
- Total records: {quality_summary['total_records']}
- Quality issues: {quality_summary['missing_value_columns']} missing value columns, {quality_summary.get('duplicate_records', 0)} duplicates
- Bias status: {bias_summary['status']}
"""
                        
                        response = ollama.generate(
                            model='llama3.2:3b',
                            prompt=f"""
You are a helpful medical data science advisor.

CONTEXT:
{context_summary}

USER QUESTION:
{user_question}

Provide a clear, specific answer in 2-4 sentences.
Use exact numbers when referencing the data.
Be helpful and educational.
""",
                            options={'temperature': 0.3}
                        )
                        
                        st.markdown("""
                        <div class="success-box">
                        <strong>🤖 AI Response:</strong>
                        </div>
                        """, unsafe_allow_html=True)
                        st.write(response['response'])
                        
                    except Exception as e:
                        st.error(f"AI chat unavailable: {e}")
                        st.info("Please check your question and try again, or ensure Ollama is running.")
            else:
                st.warning("Please enter a question first!")

# ========================================================================
# TAB 5: IMPLEMENT CHANGES
# ========================================================================

with tab5:
    st.markdown("# 🛠️ Implement Changes")
    st.caption("Select recommendations to apply and generate an improved dataset")
    
    # Get pre-computed recommendations from cache (instant - no computation delay!)
    all_recommendations = st.session_state.get('cached_recommendations', [])
    
    if not all_recommendations:
        st.markdown("""
        <div class="success-box">
        ✅ <strong>No recommendations available!</strong><br>
        Your dataset appears to be clean. No changes needed.
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="info-box">
        📋 <strong>{len(all_recommendations)} recommendations</strong> available for implementation
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        st.markdown("### Select Changes to Implement")
        
        # Initialize session state for selected recommendations
        if 'selected_recommendations' not in st.session_state:
            st.session_state.selected_recommendations = {}
        
        # Get pre-grouped recommendations from cache (no computation on rerun!)
        recommendations_by_type = st.session_state.get('cached_recommendations_grouped', {})
        
        # Display recommendations grouped by type
        for rec_type, recs in recommendations_by_type.items():
            type_labels = {
                'missing_value': '📊 Missing Values',
                'duplicate': '🔄 Duplicates',
                'outlier': '📈 Outliers',
                'bias_normalization': '⚖️ Bias Normalization',
                'bias_mitigation': '⚖️ Bias Mitigation (SMOTE/Oversampling)'
            }
            
            with st.expander(f"{type_labels.get(rec_type, rec_type)} ({len(recs)} recommendations)", expanded=True):
                for rec in recs:
                    col1, col2 = st.columns([1, 4])
                    with col1:
                        # Use Streamlit's built-in state management via key
                        # The checkbox state is automatically stored in st.session_state[f"checkbox_{rec['key']}"]
                        checkbox_key = f"checkbox_{rec['key']}"
                        selected = st.checkbox(
                            rec['method'],  # Use method name as label for accessibility
                            value=st.session_state.get(checkbox_key, False),
                            key=checkbox_key,
                            label_visibility="collapsed"  # Hide label visually but keep for accessibility
                        )
                        # Sync to our tracking dict for easy counting
                        st.session_state.selected_recommendations[rec['key']] = selected
                    
                    with col2:
                        st.markdown(f"**{rec['method']}** {rec['priority']}")
                        if rec['column']:
                            st.caption(f"Column: {rec['column']}")
                        st.write(f"💡 {rec['reason']}")
                        if rec['impact']:
                            st.caption(f"✨ Impact: {rec['impact']}")
        
        st.markdown("---")
        
        # Count selected recommendations (fast operation, no need to cache)
        selected_count = sum(1 for v in st.session_state.selected_recommendations.values() if v)
        
        col1, col2 = st.columns([3, 1])
        with col1:
            if selected_count > 0:
                st.info(f"✅ {selected_count} change(s) selected for implementation")
            else:
                st.warning("⚠️ Please select at least one change to implement")
        
        with col2:
            implement_button = st.button(
                "🚀 Implement Changes",
                use_container_width=True,
                disabled=selected_count == 0
            )
        
        # Implementation logic
        if implement_button and selected_count > 0:
            with st.spinner("⚙️ Implementing changes..."):
                try:
                    # Create a copy of the synthetic dataset
                    modified_df = synthetic_df.copy()
                    applied_changes = []
                    modified_columns = {}  # Track which columns were modified and how
                    
                    # Store original state BEFORE any modifications for accurate before/after comparison
                    original_state = {}
                    for col in synthetic_df.columns:
                        if pd.api.types.is_numeric_dtype(synthetic_df[col]):
                            original_state[col] = {
                                'missing': int(synthetic_df[col].isna().sum()),
                                'mean': float(synthetic_df[col].mean()) if not synthetic_df[col].isna().all() else None,
                                'min': float(synthetic_df[col].min()) if not synthetic_df[col].isna().all() else None,
                                'max': float(synthetic_df[col].max()) if not synthetic_df[col].isna().all() else None,
                                'std': float(synthetic_df[col].std()) if not synthetic_df[col].isna().all() else None
                            }
                        else:
                            original_state[col] = {
                                'missing': int(synthetic_df[col].isna().sum()),
                                'value_counts': synthetic_df[col].value_counts().to_dict()
                            }
                    
                    # Apply selected recommendations
                    for rec in all_recommendations:
                        if st.session_state.selected_recommendations.get(rec['key'], False):
                            try:
                                if rec['type'] == 'missing_value':
                                    # Track missing values before imputation
                                    missing_before = modified_df[rec['column']].isna().sum()
                                    
                                    # Apply imputation
                                    if 'Median' in rec['method']:
                                        value = rec.get('value')
                                        if value is None:
                                            # Calculate median if not provided
                                            value = modified_df[rec['column']].median()
                                        modified_df[rec['column']].fillna(value, inplace=True)
                                        missing_after = modified_df[rec['column']].isna().sum()
                                        applied_changes.append(f"✅ Filled {missing_before} missing values in '{rec['column']}' using {rec['method']} (value: {value:.2f})")
                                        if missing_before > 0:
                                            modified_columns[rec['column']] = {
                                                'type': 'imputation',
                                                'method': rec['method'],
                                                'values_filled': missing_before,
                                                'imputation_value': value
                                            }
                                    
                                    elif 'Mean' in rec['method']:
                                        value = rec.get('value')
                                        if value is None:
                                            # Calculate mean if not provided
                                            value = modified_df[rec['column']].mean()
                                        modified_df[rec['column']].fillna(value, inplace=True)
                                        missing_after = modified_df[rec['column']].isna().sum()
                                        applied_changes.append(f"✅ Filled {missing_before} missing values in '{rec['column']}' using {rec['method']} (value: {value:.2f})")
                                        if missing_before > 0:
                                            modified_columns[rec['column']] = {
                                                'type': 'imputation',
                                                'method': rec['method'],
                                                'values_filled': missing_before,
                                                'imputation_value': value
                                            }
                                    
                                    elif 'Mode' in rec['method']:
                                        value = rec.get('value')
                                        if value is None:
                                            # Calculate mode if not provided
                                            mode_values = modified_df[rec['column']].mode()
                                            value = mode_values[0] if len(mode_values) > 0 else None
                                        if value is not None:
                                            modified_df[rec['column']].fillna(value, inplace=True)
                                            missing_after = modified_df[rec['column']].isna().sum()
                                            applied_changes.append(f"✅ Filled {missing_before} missing values in '{rec['column']}' using {rec['method']} (value: {value})")
                                            if missing_before > 0:
                                                modified_columns[rec['column']] = {
                                                    'type': 'imputation',
                                                    'method': rec['method'],
                                                    'values_filled': missing_before,
                                                    'imputation_value': value
                                                }
                                    
                                    elif 'Remove' in rec['method']:
                                        before_count = len(modified_df)
                                        modified_df.dropna(subset=[rec['column']], inplace=True)
                                        after_count = len(modified_df)
                                        rows_removed = before_count - after_count
                                        applied_changes.append(f"✅ Removed {rows_removed} rows with missing '{rec['column']}'")
                                        if rows_removed > 0:
                                            modified_columns[rec['column']] = {
                                                'type': 'row_removal',
                                                'rows_removed': rows_removed
                                            }
                                
                                elif rec['type'] == 'duplicate':
                                    # Remove duplicates
                                    before_count = len(modified_df)
                                    modified_df.drop_duplicates(inplace=True)
                                    after_count = len(modified_df)
                                    duplicates_removed = before_count - after_count
                                    applied_changes.append(f"✅ Removed {duplicates_removed} duplicate records")
                                    if duplicates_removed > 0:
                                        modified_columns['_duplicates'] = {
                                            'type': 'duplicate_removal',
                                            'rows_removed': duplicates_removed
                                        }
                                
                                elif rec['type'] == 'outlier':
                                    # Handle outliers - for now, we'll use clipping/winsorization if recommended
                                    if 'Clip' in rec['method'] or 'Winsorize' in rec['method']:
                                        # Get bounds from quality_issues
                                        if rec['column'] in quality_issues.get('outliers', {}):
                                            bounds = quality_issues['outliers'][rec['column']]['bounds']
                                            # Count outliers before clipping
                                            outliers_before = ((modified_df[rec['column']] < bounds['lower']) | 
                                                              (modified_df[rec['column']] > bounds['upper'])).sum()
                                            modified_df[rec['column']] = modified_df[rec['column']].clip(
                                                lower=bounds['lower'],
                                                upper=bounds['upper']
                                            )
                                            applied_changes.append(f"✅ Clipped {outliers_before} outliers in '{rec['column']}' to range [{bounds['lower']:.1f}, {bounds['upper']:.1f}]")
                                            if outliers_before > 0:
                                                modified_columns[rec['column']] = {
                                                    'type': 'outlier_clipping',
                                                    'outliers_clipped': outliers_before,
                                                    'bounds': bounds
                                                }
                                
                                elif rec['type'] == 'bias_normalization':
                                    # Normalize sex/gender values
                                    original_values = modified_df[rec['column']].value_counts().to_dict()
                                    detector = BiasDetector(modified_df)
                                    normalized_series = detector._normalize_sex_gender_values(modified_df[rec['column']])
                                    modified_df[rec['column']] = normalized_series
                                    normalized_values = modified_df[rec['column']].value_counts().to_dict()
                                    applied_changes.append(f"✅ Normalized '{rec['column']}' values to standard Male/Female format")
                                    modified_columns[rec['column']] = {
                                        'type': 'normalization',
                                        'before': original_values,
                                        'after': normalized_values
                                    }
                                
                                elif rec['type'] == 'bias_mitigation':
                                    # Apply actual bias mitigation (SMOTE, oversampling, etc.)
                                    col_name = rec['column']
                                    method = rec['method']
                                    
                                    # Store original distribution
                                    original_dist = modified_df[col_name].value_counts().to_dict()
                                    original_total = len(modified_df)
                                    
                                    if method == 'SMOTE' or 'SMOTE' in method:
                                        # Apply SMOTE to balance the dataset
                                        try:
                                            from imblearn.over_sampling import SMOTE
                                            
                                            # Prepare features (X) and target (y) for SMOTE
                                            X = modified_df.drop(columns=[col_name])
                                            y = modified_df[col_name]
                                            
                                            # Check if we have numeric features for SMOTE
                                            numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
                                            categorical_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()
                                            
                                            if len(numeric_cols) > 0:
                                                # Use only numeric columns for SMOTE
                                                X_numeric = X[numeric_cols].values
                                                
                                                # Apply SMOTE
                                                smote = SMOTE(random_state=42)
                                                X_resampled, y_resampled = smote.fit_resample(X_numeric, y)
                                                
                                                # Create new dataframe with resampled numeric features
                                                resampled_df = pd.DataFrame(X_resampled, columns=numeric_cols)
                                                resampled_df[col_name] = y_resampled
                                                
                                                # Add categorical columns back
                                                # For original rows (first len(modified_df) rows), use original values
                                                # For new synthetic rows, sample from minority class
                                                for cat_col in categorical_cols:
                                                    resampled_df[cat_col] = None
                                                
                                                # Fill original rows with original categorical values
                                                for idx in range(min(len(modified_df), len(resampled_df))):
                                                    for cat_col in categorical_cols:
                                                        resampled_df.loc[idx, cat_col] = modified_df.loc[idx, cat_col]
                                                
                                                # Fill new synthetic rows with sampled minority class values
                                                if len(resampled_df) > len(modified_df):
                                                    minority_group = y_resampled[len(modified_df)]
                                                    minority_df = modified_df[modified_df[col_name] == minority_group]
                                                    
                                                    for idx in range(len(modified_df), len(resampled_df)):
                                                        # Sample a random minority row
                                                        sample_row = minority_df.sample(n=1, random_state=42 + idx).iloc[0]
                                                        for cat_col in categorical_cols:
                                                            resampled_df.loc[idx, cat_col] = sample_row[cat_col]
                                                
                                                modified_df = resampled_df
                                                
                                                new_dist = modified_df[col_name].value_counts().to_dict()
                                                new_total = len(modified_df)
                                                samples_added = new_total - original_total
                                                
                                                applied_changes.append(f"✅ Applied SMOTE to balance '{col_name}': Added {samples_added} synthetic samples")
                                                modified_columns[col_name] = {
                                                    'type': 'bias_mitigation',
                                                    'method': 'SMOTE',
                                                    'before': original_dist,
                                                    'after': new_dist,
                                                    'samples_added': samples_added
                                                }
                                            else:
                                                # Fallback to simple oversampling if no numeric features
                                                value_counts = modified_df[col_name].value_counts()
                                                majority_count = value_counts.max()
                                                minority_group = value_counts.idxmin()
                                                minority_df = modified_df[modified_df[col_name] == minority_group]
                                                samples_needed = majority_count - len(minority_df)
                                                oversampled = minority_df.sample(n=samples_needed, replace=True, random_state=42)
                                                modified_df = pd.concat([modified_df, oversampled]).reset_index(drop=True)
                                                
                                                new_dist = modified_df[col_name].value_counts().to_dict()
                                                samples_added = len(modified_df) - original_total
                                                
                                                applied_changes.append(f"✅ Applied oversampling to balance '{col_name}': Added {samples_added} samples")
                                                modified_columns[col_name] = {
                                                    'type': 'bias_mitigation',
                                                    'method': 'oversampling',
                                                    'before': original_dist,
                                                    'after': new_dist,
                                                    'samples_added': samples_added
                                                }
                                        
                                        except ImportError:
                                            applied_changes.append(f"❌ SMOTE not available. Install: pip install imbalanced-learn")
                                        except Exception as e:
                                            applied_changes.append(f"❌ SMOTE failed: {str(e)}")
                                    
                                    elif method == 'class_weighting':
                                        # Class weighting doesn't modify the dataset, just the model
                                        # So we'll skip this for dataset modification
                                        applied_changes.append(f"ℹ️ Class weighting applied (affects model training, not dataset)")
                                    
                                    elif method == 'undersampling':
                                        # Undersample majority class
                                        try:
                                            # Find majority and minority groups
                                            value_counts = modified_df[col_name].value_counts()
                                            majority_group = value_counts.idxmax()
                                            minority_group = value_counts.idxmin()
                                            minority_count = value_counts[minority_group]
                                            
                                            # Keep all minority samples, randomly sample majority to match
                                            minority_df = modified_df[modified_df[col_name] == minority_group]
                                            majority_df = modified_df[modified_df[col_name] == majority_group]
                                            majority_sampled = majority_df.sample(n=minority_count, random_state=42)
                                            
                                            modified_df = pd.concat([minority_df, majority_sampled]).reset_index(drop=True)
                                            
                                            new_dist = modified_df[col_name].value_counts().to_dict()
                                            rows_removed = original_total - len(modified_df)
                                            
                                            applied_changes.append(f"✅ Applied undersampling to '{col_name}': Removed {rows_removed} majority samples")
                                            modified_columns[col_name] = {
                                                'type': 'bias_mitigation',
                                                'method': 'undersampling',
                                                'before': original_dist,
                                                'after': new_dist,
                                                'rows_removed': rows_removed
                                            }
                                        
                                        except Exception as e:
                                            applied_changes.append(f"❌ Undersampling failed: {str(e)}")
                            
                            except Exception as e:
                                applied_changes.append(f"❌ Failed to apply {rec['method']}: {str(e)}")
                    
                    # Store modified dataset in session state
                    st.session_state.modified_dataset = modified_df
                    st.session_state.applied_changes = applied_changes
                    
                    st.success(f"✅ Successfully applied {len([c for c in applied_changes if c.startswith('✅')])} change(s)!")
                    
                    # Show applied changes
                    st.markdown("### 📋 Applied Changes:")
                    for change in applied_changes:
                        if change.startswith('✅'):
                            st.success(change)
                        else:
                            st.error(change)
                    
                    # Show dataset summary
                    st.markdown("---")
                    st.markdown("### 📊 Modified Dataset Summary:")
                    
                    # Calculate actual changes
                    original_count = len(synthetic_df)
                    new_count = len(modified_df)
                    delta = new_count - original_count
                    total_values_changed = sum(col_info.get('values_filled', 0) + col_info.get('outliers_clipped', 0) 
                                             for col_info in modified_columns.values() if isinstance(col_info, dict))
                    
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Total Records", f"{len(modified_df):,}", delta=f"{delta:+,}" if delta != 0 else None)
                    with col2:
                        st.metric("Features", len(modified_df.columns))
                    with col3:
                        st.metric("Columns Modified", len([k for k in modified_columns.keys() if k != '_duplicates']))
                    with col4:
                        if total_values_changed > 0:
                            st.metric("Values Changed", f"{total_values_changed:,}", delta=f"+{total_values_changed:,}")
                        else:
                            st.metric("Values Changed", "0")
                    
                    # Show detailed changes if any columns were modified
                    if modified_columns:
                        st.markdown("---")
                        st.markdown("### 🔍 Detailed Changes:")
                        
                        for col_name, col_info in modified_columns.items():
                            if col_name == '_duplicates':
                                st.markdown(f"**🔄 Duplicate Records:**")
                                st.write(f"  • Removed {col_info['rows_removed']} duplicate rows")
                            elif col_info['type'] == 'imputation':
                                st.markdown(f"**📊 Column: `{col_name}`**")
                                st.write(f"  • **Action:** Filled {col_info['values_filled']} missing values")
                                st.write(f"  • **Method:** {col_info['method']}")
                                st.write(f"  • **Imputation Value:** {col_info['imputation_value']}")
                                
                                # Show before/after stats
                                col_before, col_after = st.columns(2)
                                with col_before:
                                    st.caption("**Before:**")
                                    missing_before = synthetic_df[col_name].isna().sum()
                                    st.write(f"  • Missing: {missing_before}")
                                    if not synthetic_df[col_name].isna().all():
                                        st.write(f"  • Mean: {synthetic_df[col_name].mean():.2f}")
                                with col_after:
                                    st.caption("**After:**")
                                    missing_after = modified_df[col_name].isna().sum()
                                    st.write(f"  • Missing: {missing_after}")
                                    st.write(f"  • Mean: {modified_df[col_name].mean():.2f}")
                            
                            elif col_info['type'] == 'outlier_clipping':
                                st.markdown(f"**📈 Column: `{col_name}`**")
                                st.write(f"  • **Action:** Clipped {col_info['outliers_clipped']} outliers")
                                st.write(f"  • **Range:** [{col_info['bounds']['lower']:.1f}, {col_info['bounds']['upper']:.1f}]")
                                
                                col_before, col_after = st.columns(2)
                                with col_before:
                                    st.caption("**Before:**")
                                    orig = original_state.get(col_name, {})
                                    if orig.get('min') is not None:
                                        st.write(f"  • Min: {orig['min']:.2f}")
                                        st.write(f"  • Max: {orig['max']:.2f}")
                                        st.write(f"  • Mean: {orig.get('mean', 0):.2f}")
                                with col_after:
                                    st.caption("**After:**")
                                    min_after = modified_df[col_name].min()
                                    max_after = modified_df[col_name].max()
                                    mean_after = modified_df[col_name].mean()
                                    st.write(f"  • Min: {min_after:.2f} {'✅' if orig.get('min') and min_after != orig['min'] else ''}")
                                    st.write(f"  • Max: {max_after:.2f} {'✅' if orig.get('max') and max_after != orig['max'] else ''}")
                                    st.write(f"  • Mean: {mean_after:.2f}")
                                    st.write(f"  • **Outliers Clipped:** {col_info['outliers_clipped']}")
                            
                            elif col_info['type'] == 'normalization':
                                st.markdown(f"**⚖️ Column: `{col_name}`**")
                                st.write(f"  • **Action:** Normalized sex/gender values to standard Male/Female format")
                                
                                col_before, col_after = st.columns(2)
                                with col_before:
                                    st.caption("**Before (Mixed Encodings):**")
                                    orig = original_state.get(col_name, {})
                                    if 'value_counts' in orig:
                                        total_before = sum(orig['value_counts'].values())
                                        for val, count in sorted(orig['value_counts'].items(), key=lambda x: x[1], reverse=True):
                                            pct = (count / total_before) * 100 if total_before > 0 else 0
                                            st.write(f"  • `{val}`: {count} ({pct:.1f}%)")
                                    elif 'before' in col_info:
                                        total_before = sum(col_info['before'].values())
                                        for val, count in sorted(col_info['before'].items(), key=lambda x: x[1], reverse=True):
                                            pct = (count / total_before) * 100 if total_before > 0 else 0
                                            st.write(f"  • `{val}`: {count} ({pct:.1f}%)")
                                with col_after:
                                    st.caption("**After (Standardized):**")
                                    after_counts = modified_df[col_name].value_counts().to_dict()
                                    total_after = sum(after_counts.values())
                                    for val, count in sorted(after_counts.items(), key=lambda x: x[1], reverse=True):
                                        pct = (count / total_after) * 100 if total_after > 0 else 0
                                        st.write(f"  • `{val}`: {count} ({pct:.1f}%) {'✅' if val in ['Male', 'Female'] else ''}")
                                    # Show how many unique values were normalized
                                    if 'before' in col_info:
                                        before_unique = len(col_info['before'])
                                        after_unique = len(after_counts)
                                        if before_unique > after_unique:
                                            st.success(f"  **Normalized:** {before_unique} → {after_unique} unique values")
                                
                            elif col_info['type'] == 'bias_mitigation':
                                st.markdown(f"**⚖️ Column: `{col_name}`**")
                                method = col_info.get('method', 'Unknown')
                                st.write(f"  • **Action:** Applied {method} to balance distribution")
                                
                                if 'samples_added' in col_info:
                                    st.write(f"  • **Samples Added:** {col_info['samples_added']:,} synthetic samples")
                                elif 'rows_removed' in col_info:
                                    st.write(f"  • **Rows Removed:** {col_info['rows_removed']:,} majority samples")
                                
                                col_before, col_after = st.columns(2)
                                with col_before:
                                    st.caption("**Before (Imbalanced):**")
                                    if 'before' in col_info:
                                        total_before = sum(col_info['before'].values())
                                        for val, count in sorted(col_info['before'].items(), key=lambda x: x[1], reverse=True):
                                            pct = (count / total_before) * 100 if total_before > 0 else 0
                                            st.write(f"  • `{val}`: {count:,} ({pct:.1f}%)")
                                
                                with col_after:
                                    st.caption("**After (Balanced):**")
                                    if 'after' in col_info:
                                        total_after = sum(col_info['after'].values())
                                        for val, count in sorted(col_info['after'].items(), key=lambda x: x[1], reverse=True):
                                            pct = (count / total_after) * 100 if total_after > 0 else 0
                                            # Show improvement indicator
                                            if 'before' in col_info and val in col_info['before']:
                                                before_pct = (col_info['before'][val] / sum(col_info['before'].values())) * 100
                                                improvement = pct - before_pct
                                                indicator = "✅" if abs(pct - 50) < abs(before_pct - 50) else ""
                                                st.write(f"  • `{val}`: {count:,} ({pct:.1f}%) {indicator} {f'({improvement:+.1f}%)' if abs(improvement) > 0.1 else ''}")
                                            else:
                                                st.write(f"  • `{val}`: {count:,} ({pct:.1f}%) ✅")
                                    
                                    # Calculate balance improvement
                                    if 'before' in col_info and 'after' in col_info:
                                        before_values = list(col_info['before'].values())
                                        after_values = list(col_info['after'].values())
                                        if len(before_values) >= 2 and len(after_values) >= 2:
                                            before_gap = max(before_values) - min(before_values)
                                            after_gap = max(after_values) - min(after_values)
                                            gap_reduction = before_gap - after_gap
                                            if gap_reduction > 0:
                                                st.success(f"  **Balance Gap Reduced:** {gap_reduction:.1f}%")
                                
                                st.markdown("---")
                            
                            st.markdown("---")
                    else:
                        st.info("ℹ️ No data values were modified. Selected changes may have been redundant or already applied.")
                    
                    # Download button
                    st.markdown("---")
                    csv_data = modified_df.to_csv(index=False)
                    st.download_button(
                        label="💾 Download Modified Dataset (CSV)",
                        data=csv_data,
                        file_name=f"improved_dataset_{len(modified_df)}_records.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
                    
                    # Show sample of modified data
                    with st.expander("📊 Preview Modified Dataset", expanded=False):
                        st.dataframe(modified_df.head(10), use_container_width=True)
                
                except Exception as e:
                    st.error(f"❌ Error implementing changes: {str(e)}")
                    st.exception(e)

st.session_state.analysis_complete = True

# ============================================================================
# FOOTER
# ============================================================================

st.divider()

footer_col1, footer_col2, footer_col3 = st.columns(3)
with footer_col1:
    st.markdown("<div style='text-align: center;'>🏥 <strong>MedGuard AI</strong></div>", unsafe_allow_html=True)
with footer_col2:
    st.markdown("<div style='text-align: center;'>🏆 <strong>Convergence 2026 Hackathon</strong></div>", unsafe_allow_html=True)
with footer_col3:
    st.markdown("<div style='text-align: center;'>👥 <strong>The Guardians Team</strong></div>", unsafe_allow_html=True)

st.write("")
st.markdown("<div style='text-align: center; opacity: 0.5;'><small>Built with ❤️ using Streamlit • Powered by Llama 3.2</small></div>", unsafe_allow_html=True)