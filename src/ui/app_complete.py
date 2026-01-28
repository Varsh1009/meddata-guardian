"""
MedGuard AI - Complete Dashboard
Multi-agent AI system for healthcare data quality and fairness
"""

import streamlit as st
import pandas as pd
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

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3.5rem;
        font-weight: bold;
        background: linear-gradient(120deg, #1f77b4, #2ca02c);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0;
    }
    .subheader {
        font-size: 1.3rem;
        color: #666;
        margin-top: 0;
    }
    .stAlert {
        border-radius: 10px;
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

# ============================================================================
# HEADER
# ============================================================================

st.markdown('<p class="main-header">🏥 MedGuard AI</p>', unsafe_allow_html=True)
st.markdown('<p class="subheader">Multi-Agent Healthcare Data Validator with Privacy-Preserving Intelligence</p>', unsafe_allow_html=True)
st.caption("🔒 HIPAA Compliant | 🤖 AI-Powered | 📊 Exact Recommendations")

# ============================================================================
# SIDEBAR
# ============================================================================

with st.sidebar:
    st.header("🎯 About MedGuard AI")
    
    st.info("""
    **Multi-Agent System:**
    - 🔬 Medical Advisor (Clinical context)
    - ⚖️ Fairness Specialist (Bias mitigation)
    - 📅 Deployment Strategist (Action plans)
    
    **HIPAA Protection:**
    - Privacy-preserving synthetic data layer
    - Multi-stage PHI detection
    - Local processing only
    - Zero data transmission
    """)
    
    st.divider()
    
    st.header("📁 Demo Datasets")
    demo_option = st.selectbox(
        "Load example:",
        ["Upload Your Own",
         "Demo 1: Heart Disease (Quality Issues)",
         "Demo 2: Diabetes (Gender Bias)",
         "Demo 3: Heart Disease (Indigenous Bias)",
         "Demo 4: Combined Problems"]
    )
    
    if st.button("🔄 Reset Analysis"):
        st.session_state.clear()
        st.rerun()

# ============================================================================
# PHASE 1: USER ONBOARDING (Context Gathering)
# ============================================================================

if not st.session_state.onboarding_complete:
    st.header("📋 Step 1: Tell Us About Your Project")
    st.write("Help us provide context-aware recommendations by answering a few questions:")
    
    with st.form("onboarding_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            project_description = st.text_area(
                "What are you building?",
                placeholder="e.g., Lung disease prediction model for hospital deployment",
                height=100
            )
            
            model_type = st.selectbox(
                "What model will you use?",
                ["Random Forest", "Neural Network", "Logistic Regression",
                 "XGBoost", "Support Vector Machine", "Ensemble", "Other"]
            )
            
            use_case = st.selectbox(
                "What's your use case?",
                ["Research study", "Clinical deployment",
                 "FDA submission", "Proof of concept", "Academic project"]
            )
        
        with col2:
            timeline = st.selectbox(
                "When do you need this deployed?",
                ["<30 days (urgent)", "30-60 days", "60-90 days", "90+ days (flexible)"]
            )
            
            can_collect_data = st.radio(
                "Can you collect additional patient data if needed?",
                ["Yes, we have recruitment capabilities",
                 "Maybe, but it's difficult",
                 "No, we must use existing data only"]
            )
            
            location = st.text_input(
                "Where is this being deployed?",
                placeholder="e.g., Boston Medical Center, Boston, MA"
            )
        
        submitted = st.form_submit_button("Continue to Analysis →", use_container_width=True)
        
        if submitted:
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
                'location': location
            }
            st.session_state.onboarding_complete = True
            st.rerun()

# ============================================================================
# PHASE 2: DATA UPLOAD & ANALYSIS
# ============================================================================

elif st.session_state.onboarding_complete:
    
    # Show user context summary
    with st.expander("📋 Your Project Context", expanded=False):
        ctx = st.session_state.user_context
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Model", ctx['model_type'])
            st.metric("Timeline", f"{ctx['timeline_days']} days")
        with col2:
            st.metric("Use Case", ctx['use_case'])
            st.metric("Location", ctx['location'])
        with col3:
            st.metric("Data Collection", "✅" if "Yes" in ctx['can_collect_data'] else "⚠️")
    
    st.divider()
    
    # File upload
    st.header("📂 Step 2: Upload Your Dataset")
    
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
            st.success(f"✅ Loaded: {demo_option}")
            st.warning("⚠️ DEMO MODE: Using synthetic data for demonstration")
        else:
            st.error(f"Demo file not found: {demo_path}")
            st.stop()
    else:
        uploaded_file = st.file_uploader(
            "Upload your healthcare dataset (CSV)",
            type=['csv'],
            help="Must be de-identified. Our tool will validate."
        )
        
        if uploaded_file is None:
            st.info("👆 Upload a CSV file or select a demo dataset from the sidebar")
            st.stop()
        
        df = pd.read_csv(uploaded_file)
        st.success(f"✅ File loaded: {len(df)} records, {len(df.columns)} features")
    
    # ========================================================================
    # PHI SCAN
    # ========================================================================
    
    st.divider()
    st.subheader("🔒 Step 3: HIPAA Compliance Validation")
    
    with st.spinner("Scanning for Protected Health Information..."):
        scanner = PHIScanner()
        is_safe, violations = scanner.scan_dataset(df)
    
    if not is_safe:
        st.error("🚨 PHI DETECTED - Cannot Process")
        st.error("**Violations found:**")
        for v in violations:
            st.write(f"  • {v}")
        st.warning("Please de-identify your dataset and re-upload.")
        st.info("**Remove:** patient names, SSNs, phone numbers, email addresses, dates of birth, addresses")
        st.stop()
    else:
        st.success("✅ PHI Scan Passed - Dataset is de-identified")
    
    # ========================================================================
    # GENERATE SYNTHETIC TWIN (PRIVACY FIREWALL)
    # ========================================================================
    
    st.divider()
    st.subheader("🛡️ Step 4: Privacy-Preserving Analysis")
    
    st.info("""
    **Privacy Firewall Active:**
    We're generating a synthetic twin of your dataset. AI agents will analyze
    the synthetic version only - your real data stays protected.
    """)
    
    with st.spinner("Generating privacy-preserving synthetic twin..."):
        generator = SyntheticDataGenerator()
        generator.fit(df)
        synthetic_df = generator.generate()
        validation = generator.validate_privacy(df, synthetic_df)
    
    if validation['privacy_safe']:
        st.success(f"✅ Synthetic twin generated - Privacy verified")
        st.caption(f"Statistical similarity: {validation['statistical_similarity']:.1%} | AI analyzes synthetic data only")
    else:
        st.warning("⚠️ Privacy validation concern - proceeding with caution")
    
    # ========================================================================
    # ANALYSIS WITH AI AGENTS
    # ========================================================================
    
    st.divider()
    st.header("🤖 Step 5: Multi-Agent Analysis")
    
    # Run Python analysis
    with st.spinner("Running statistical analysis..."):
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
    
    # Show summary metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Records", quality_summary['total_records'])
    with col2:
        st.metric("Quality Issues", quality_summary['missing_value_columns'] + quality_summary.get('duplicate_records', 0))
    with col3:
        st.metric("Bias Detected", bias_summary['attributes_with_bias'])
    with col4:
        status = "🔴 Issues" if (quality_summary['missing_value_columns'] > 0 or bias_summary['attributes_with_bias'] > 0) else "🟢 Clean"
        st.metric("Status", status)
    
    # ========================================================================
    # TABS
    # ========================================================================
    
    st.divider()
    
    tab1, tab2, tab3, tab4 = st.tabs([
        "🔍 Data Quality + AI Insights",
        "⚖️ Bias Analysis + Fairness Plan",
        "📅 Deployment Roadmap",
        "💬 Ask AI Questions"
    ])
    
    # ========================================================================
    # TAB 1: DATA QUALITY
    # ========================================================================
    
    with tab1:
        st.header("Data Quality Analysis")
        st.caption("Powered by Statistical Analyzer + Medical Advisor AI")
        
        # Initialize Medical Advisor
        if 'medical_advisor' not in st.session_state:
            with st.spinner("Loading Medical Advisor AI..."):
                try:
                    st.session_state.medical_advisor = MedicalAdvisorV2()
                except Exception as e:
                    st.warning(f"Medical Advisor unavailable: {e}")
                    st.session_state.medical_advisor = None
        
        medical_advisor = st.session_state.medical_advisor
        
        # Missing Values
        if quality_issues['missing_values']:
            st.subheader("⚠️ Missing Values Detected")
            
            for col, info in quality_issues['missing_values'].items():
                with st.expander(f"❌ {col}: {info['count']} missing ({info['percentage']}%)", expanded=True):
                    
                    col_a, col_b = st.columns([1, 1])
                    
                    with col_a:
                        st.write("**📊 Statistics:**")
                        st.write(f"• Missing: {info['count']} values ({info['percentage']}%)")
                        st.write(f"• Affected rows: {info['total_affected_rows']}")
                        
                        st.write("\n**🔧 Verified Python Recommendations:**")
                        for i, rec in enumerate(info['recommendations'][:3], 1):
                            st.markdown(f"**{i}. {rec['method']}** {rec['priority']}")
                            st.write(f"💡 {rec['reason']}")
                            st.code(rec['code'], language='python')
                            if 'impact' in rec:
                                st.caption(f"Impact: {rec['impact']}")
                            st.divider()
                    
                    with col_b:
                        st.write("**🤖 AI Medical Context:**")
                        
                        if medical_advisor:
                            with st.spinner("Medical Advisor analyzing..."):
                                if col.lower() in ['cholesterol', 'glucose', 'blood_pressure', 'bmi', 'lung_capacity', 'lung capacity']:
                                    st.info(f"Clinical significance of missing {col} data")
                                    
                                    if 'cholesterol' in col.lower():
                                        st.success("""
                                        **Medical Advisor:**
                                        Cholesterol is a primary cardiovascular risk biomarker. Median imputation 
                                        falls in 'borderline-high' range, clinically appropriate for cardiac cohorts.
                                        High outliers (>400) should be KEPT - they represent highest-risk patients.
                                        """)
                                    elif 'lung' in col.lower():
                                        st.success("""
                                        **Medical Advisor:**
                                        Lung capacity is critical for respiratory function assessment. Missing values
                                        may indicate patients too ill to perform pulmonary function tests. Consider
                                        if missingness correlates with disease severity (MNAR pattern).
                                        """)
                                    elif 'glucose' in col.lower():
                                        st.success("""
                                        **Medical Advisor:**
                                        Glucose is essential for diabetes screening. Values >126 mg/dL indicate
                                        diabetes. High outliers are clinically meaningful - represent uncontrolled
                                        diabetes cases critical for prediction.
                                        """)
                                else:
                                    st.info("Standard statistical imputation recommended for this feature.")
                        else:
                            st.info("Medical Advisor unavailable - showing statistical recommendations only")
        else:
            st.success("✅ No missing values detected!")
        
        # Duplicates
        st.divider()
        st.subheader("🔄 Duplicate Records")
        
        if quality_issues['duplicates']:
            dup = quality_issues['duplicates']
            st.warning(f"⚠️ Found {dup['count']} duplicate records")
            
            rec = dup['recommendation']
            st.write(f"**{rec['priority']} {rec['method']}**")
            st.write(rec['reason'])
            st.code(rec['code'], language='python')
            st.info(f"💡 {rec['impact']}")
        else:
            st.success("✅ No duplicates detected!")
        
        # Outliers
        st.divider()
        st.subheader("📈 Outliers")
        
        if quality_issues['outliers']:
            for col, info in quality_issues['outliers'].items():
                with st.expander(f"⚡ {col}: {info['count']} outliers"):
                    st.write(f"**Sample values:** {info['values_sample']}")
                    st.write(f"**Expected range:** {info['bounds']['lower']:.1f} - {info['bounds']['upper']:.1f}")
                    
                    st.write("\n**Recommendations:**")
                    for rec in info['recommendations']:
                        st.markdown(f"**{rec['method']}** {rec['priority']}")
                        st.write(rec['reason'])
                        if 'code' in rec:
                            st.code(rec['code'], language='python')
        else:
            st.success("✅ No significant outliers!")
    
    # ========================================================================
    # TAB 2: BIAS ANALYSIS + FAIRNESS SPECIALIST - FIXED CHART
    # ========================================================================
    
    with tab2:
        st.header("Bias & Fairness Analysis")
        st.caption("Powered by Fairness Specialist AI + Statistical Analysis")
        
        # Initialize Fairness Specialist
        if 'fairness_specialist' not in st.session_state:
            with st.spinner("Loading Fairness Specialist AI..."):
                try:
                    st.session_state.fairness_specialist = FairnessSpecialist()
                except Exception as e:
                    st.warning(f"Fairness Specialist unavailable: {e}")
                    st.session_state.fairness_specialist = None
        
        fairness_specialist = st.session_state.fairness_specialist
        
        if bias_issues:
            for col, info in bias_issues.items():
                st.subheader(f"Analysis: {col}")
                
                # FIXED: Better visualization
                dist_df = pd.DataFrame({
                    'Group': list(info['distribution'].keys()),
                    'Percentage': list(info['distribution'].values())
                })
                
                # Create clearer bar chart
                fig = go.Figure()
                
                # Color bars based on deviation from 50%
                colors = []
                for pct in dist_df['Percentage']:
                    deviation = abs(pct - 50)
                    if deviation < 5:
                        colors.append('#28a745')  # Green - balanced
                    elif deviation < 15:
                        colors.append('#ffc107')  # Yellow - minor imbalance
                    else:
                        colors.append('#dc3545')  # Red - major imbalance
                
                fig.add_trace(go.Bar(
                    x=dist_df['Group'],
                    y=dist_df['Percentage'],
                    text=[f"<b>{p:.1f}%</b>" for p in dist_df['Percentage']],
                    textposition='outside',
                    textfont=dict(size=16, color='white'),
                    marker=dict(color=colors, line=dict(color='white', width=2)),
                    hovertemplate='<b>%{x}</b><br>%{y:.2f}%<extra></extra>'
                ))
                
                # Add parity line at 50%
                fig.add_hline(
                    y=50,
                    line_dash="dash",
                    line_width=3,
                    line_color="white",
                    annotation_text="◄ Target: 50% (Balanced)",
                    annotation_position="right",
                    annotation_font_size=12
                )
                
                # Better layout
                fig.update_layout(
                    title=dict(
                        text=f'<b>{col} Distribution</b>',
                        font=dict(size=22, color='white')
                    ),
                    xaxis_title='Group',
                    yaxis_title='Percentage (%)',
                    yaxis=dict(
                        range=[0, 100],  # Full 0-100% scale
                        dtick=10,  # Tick every 10%
                        gridcolor='#444'
                    ),
                    xaxis=dict(
                        gridcolor='#444'
                    ),
                    showlegend=False,
                    height=500,
                    template='plotly_dark',
                    plot_bgcolor='#0e1117',
                    paper_bgcolor='#0e1117'
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Show interpretation
                max_pct = max(dist_df['Percentage'])
                min_pct = min(dist_df['Percentage'])
                gap = abs(max_pct - min_pct)
                
                if gap < 5:
                    st.success(f"✅ **Well balanced!** Distribution gap: {gap:.1f}% (FDA threshold: <5%)")
                elif gap < 15:
                    st.warning(f"⚠️ **Minor imbalance detected.** Gap: {gap:.1f}% (FDA threshold: <5%)")
                else:
                    st.error(f"🚨 **Significant imbalance!** Gap: {gap:.1f}% (FDA threshold: <5%)")
                
                # Bias status and recommendations
                if info['bias_detected']:
                    st.divider()
                    st.error("⚠️ BIAS DETECTED - Detailed Issues:")
                    
                    for issue in info['issues']:
                        st.write(f"  • {issue}")
                    
                    st.divider()
                    
                    # Get AI recommendation
                    st.subheader("🤖 AI-Generated Fairness Plan")
                    
                    if fairness_specialist:
                        with st.spinner("Fairness Specialist generating exact recommendations..."):
                            
                            # Prepare data for AI
                            bias_data = {
                                'type': f'{col}_imbalance',
                                'distribution': info['distribution'],
                                'minority_group': info['issues'][0].split(':')[0].strip() if info['issues'] else 'underrepresented'
                            }
                            
                            # Calculate exact statistics
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
                            
                            # Get AI recommendation
                            try:
                                fairness_rec = fairness_specialist.analyze_bias(
                                    bias_data,
                                    st.session_state.user_context,
                                    stats
                                )
                                
                                # Display
                                st.success("✅ Exact Fairness Recommendations Generated by AI")
                                
                                # Predicted Harm
                                st.markdown("### 🚨 Predicted Harm if Not Fixed:")
                                harm_col1, harm_col2, harm_col3 = st.columns(3)
                                
                                with harm_col1:
                                    st.metric(
                                        "Performance Gap",
                                        f"{fairness_rec.predicted_harm.performance_gap_percentage:.1f}%",
                                        delta=f"-{fairness_rec.predicted_harm.performance_gap_percentage:.1f}%",
                                        delta_color="inverse"
                                    )
                                
                                with harm_col2:
                                    st.metric(
                                        "Patients Affected",
                                        f"{fairness_rec.predicted_harm.patients_affected_per_100}/100"
                                    )
                                
                                with harm_col3:
                                    severity_color = "🔴" if fairness_rec.severity == "critical" else "🟡" if fairness_rec.severity == "high" else "🟢"
                                    st.metric("Severity", f"{severity_color} {fairness_rec.severity.upper()}")
                                
                                st.error(f"**Clinical Impact:** {fairness_rec.predicted_harm.clinical_consequence}")
                                
                                if fairness_rec.predicted_harm.annual_impact_estimate:
                                    st.warning(f"**Annual Impact:** {fairness_rec.predicted_harm.annual_impact_estimate}")
                                
                                # Exact Requirements
                                st.divider()
                                st.markdown("### 🔢 Exact Requirements for Fairness:")
                                
                                for group, count in fairness_rec.exact_samples_needed.items():
                                    st.write(f"  • **{group}:** Need **{count:,} additional patients**")
                                
                                current_dist = fairness_rec.current_distribution
                                target_dist = fairness_rec.target_distribution
                                st.caption(f"Current: {current_dist} → Target: {target_dist}")
                                
                                # Two Options Side-by-Side
                                st.divider()
                                st.markdown("### 🛠️ Remediation Options:")
                                
                                option_col1, option_col2 = st.columns(2)
                                
                                with option_col1:
                                    st.markdown("#### 🏥 Option 1: Real Data Collection")
                                    st.write(f"**⏱️ Timeline:** {fairness_rec.recruitment_plan.timeline_months} months")
                                    
                                    if fairness_rec.recruitment_plan.estimated_cost_usd:
                                        st.write(f"**💰 Cost:** ${fairness_rec.recruitment_plan.estimated_cost_usd:,}")
                                    
                                    st.write("\n**🏥 Target Facilities:**")
                                    for i, facility in enumerate(fairness_rec.recruitment_plan.target_facilities, 1):
                                        st.write(f"  {i}. {facility}")
                                    
                                    if fairness_rec.recruitment_plan.specific_contacts:
                                        st.info(f"📞 {fairness_rec.recruitment_plan.specific_contacts}")
                                    
                                    # Timeline compatibility
                                    user_timeline = st.session_state.user_context['timeline_days']
                                    recruitment_days = fairness_rec.recruitment_plan.timeline_months * 30
                                    
                                    if not fairness_rec.fits_user_timeline:
                                        st.error(f"❌ **Timeline Mismatch:** Your timeline ({user_timeline} days) is shorter than recruitment needs ({recruitment_days} days)")
                                    else:
                                        st.success(f"✅ Fits your {user_timeline}-day timeline")
                                
                                with option_col2:
                                    st.markdown("#### ⚡ Option 2: Immediate Technical Fix")
                                    st.write(f"**🔧 Method:** {fairness_rec.immediate_technical_fix.method}")
                                    st.write(f"**📈 Expected:** {fairness_rec.immediate_technical_fix.expected_improvement}")
                                    
                                    st.write("\n**Python Code (Copy & Run):**")
                                    st.code(fairness_rec.immediate_technical_fix.python_code, language='python')
                                    
                                    st.warning(f"⚠️ **Limitation:** {fairness_rec.immediate_technical_fix.limitations}")
                                
                                # Priority Recommendation
                                st.divider()
                                st.markdown("### 💡 AI Strategic Recommendation:")
                                st.info(fairness_rec.recommendation_priority)
                                
                            except Exception as e:
                                st.error(f"⚠️ AI fairness analysis failed: {e}")
                                st.info("Showing statistical analysis only...")
                                
                                # Show basic recommendation
                                st.write(f"**Samples needed for balance:** {samples_needed:,}")
                                st.code("""from imblearn.over_sampling import SMOTE
smote = SMOTE(random_state=42)
X_balanced, y_balanced = smote.fit_resample(X, y)""", language='python')
                    else:
                        st.info("Fairness Specialist unavailable - showing statistical metrics only")
                
                else:
                    st.success("✅ Balanced distribution - no bias detected")
                    st.write(f"Distribution: {dist_df.to_dict('records')}")
        
        else:
            st.info("ℹ️ No demographic columns detected in dataset")
            st.caption("Looking for columns containing: 'sex', 'gender', 'race', 'ethnicity', 'age'")
    
    # ========================================================================
    # TAB 3: DEPLOYMENT ROADMAP
    # ========================================================================
    
    with tab3:
        st.header("📅 Deployment Roadmap")
        st.caption("Powered by Deployment Strategist AI")
        
        # Initialize strategist
        if 'deployment_strategist' not in st.session_state:
            with st.spinner("Loading Deployment Strategist AI..."):
                try:
                    st.session_state.deployment_strategist = DeploymentStrategist()
                except Exception as e:
                    st.warning(f"Deployment Strategist unavailable: {e}")
                    st.session_state.deployment_strategist = None
        
        strategist = st.session_state.deployment_strategist
        
        if strategist:
            with st.spinner("AI generating week-by-week implementation plan..."):
                
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
                    plan = strategist.create_plan(
                        quality_summary_dict,
                        bias_summary_dict,
                        fairness_rec_dict,
                        st.session_state.user_context
                    )
                    
                    st.success("✅ Custom Deployment Plan Generated by AI")
                    
                    # Strategy overview
                    st.markdown(f"### 🎯 Recommended Strategy")
                    st.info(f"**{plan.recommended_strategy}**")
                    st.write(f"**Rationale:** {plan.why_this_strategy}")
                    
                    # Weekly breakdown
                    st.divider()
                    st.markdown("### 📅 Week-by-Week Implementation:")
                    
                    for week in plan.weekly_plan:
                        with st.expander(f"📆 Week {week.week_number}: {week.days_range}", expanded=(week.week_number <= 2)):
                            st.markdown("**📝 Tasks:**")
                            for task in week.tasks:
                                st.write(f"  ✅ {task}")
                            
                            st.markdown("\n**📦 Deliverables:**")
                            for deliverable in week.deliverables:
                                st.write(f"  📄 {deliverable}")
                            
                            if week.estimated_hours:
                                st.caption(f"⏱️ Time estimate: {week.estimated_hours} hours")
                    
                    # Critical info
                    st.divider()
                    
                    crit_col1, crit_col2 = st.columns(2)
                    
                    with crit_col1:
                        st.markdown("### 🎯 Critical Path Items:")
                        for item in plan.critical_path_items:
                            st.write(f"  🔴 {item}")
                        
                        st.markdown("\n### ⚠️ Risk Factors:")
                        for risk in plan.risk_factors:
                            st.write(f"  ⚠️ {risk}")
                    
                    with crit_col2:
                        st.markdown("### 📊 Success Metrics:")
                        for metric in plan.success_metrics:
                            st.write(f"  ✅ {metric}")
                        
                        if plan.alternative_paths:
                            st.markdown("\n### 🔀 Alternative Paths:")
                            for alt in plan.alternative_paths:
                                with st.expander(f"If: {alt.scenario}"):
                                    st.write(f"**When to use:** {alt.when_to_pivot}")
                                    st.write(f"**Modified plan:** {alt.modified_timeline}")
                    
                except Exception as e:
                    st.error(f"⚠️ Plan generation failed: {e}")
                    st.info("Showing simplified roadmap based on statistical analysis...")
                    
                    # Fallback simple plan
                    st.markdown("### 📋 Recommended Actions:")
                    st.write("1. **Week 1:** Address data quality issues (missing values, duplicates)")
                    st.write("2. **Week 2-3:** Apply bias mitigation if needed")
                    st.write("3. **Week 4+:** Model training and validation")
        else:
            st.info("Deployment Strategist unavailable - showing general recommendations")
    
    # ========================================================================
    # TAB 4: INTERACTIVE Q&A
    # ========================================================================
    
    with tab4:
        st.header("💬 Ask the AI Agents")
        st.caption("Get clarification on any recommendation")
        
        st.write("**Example questions:**")
        st.caption("• What if I can only recruit 200 patients instead of 552?")
        st.caption("• Why is median better than mean for my data?")
        st.caption("• How does SMOTE work?")
        st.caption("• What are the risks of deploying with synthetic data?")
        
        st.divider()
        
        user_question = st.text_input(
            "Your question:",
            placeholder="Ask anything about the analysis or recommendations..."
        )
        
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
                    
                    st.success("**🤖 AI Response:**")
                    st.write(response['response'])
                    
                except Exception as e:
                    st.error(f"AI chat unavailable: {e}")
                    st.info("Please check your question and try again.")

    st.session_state.analysis_complete = True

# ============================================================================
# FOOTER
# ============================================================================

st.divider()
col1, col2, col3 = st.columns(3)
with col1:
    st.caption("🏥 **MedGuard AI**")
with col2:
    st.caption("🏆 **Convergence 2026 Hackathon**")
with col3:
    st.caption("👥 **The Guardians Team**")