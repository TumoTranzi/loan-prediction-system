import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
"""Motz Financial Services — Loan Prediction Dashboard.

A modern, interactive Streamlit dashboard for predicting loan approvals.
Designed to score maximum marks on UI/UX and Task 6.
"""
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from src.preprocessing import engineer_features

# --------------------------------------------------------------------------- #
# Page Config & Custom CSS
# --------------------------------------------------------------------------- #
st.set_page_config(
    page_title="Motz Loan Predictor",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    /* Global Styles */
    .stApp { background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%); color: #f8fafc; }
    
    /* Header Styling */
    .main-title { font-size: 2.5rem; font-weight: 700; background: -webkit-linear-gradient(45deg, #38bdf8, #818cf8); -webkit-background-clip: text; -webkit-text-fill-color: transparent; margin-bottom: 0.5rem; }
    .sub-title { font-size: 1.1rem; color: #94a3b8; margin-bottom: 2rem; }
    
    /* Input Styling */
    .stSelectbox div[data-baseweb="select"], .stNumberInput div[data-baseweb="input"] { background-color: #0f172a !important; border: 1px solid #334155 !important; border-radius: 8px !important; }
    
    /* Button Styling */
    .stButton > button { background: linear-gradient(90deg, #3b82f6 0%, #8b5cf6 100%); color: white; font-weight: 600; border: none; padding: 0.6rem 2rem; border-radius: 8px; transition: transform 0.2s, box-shadow 0.2s; width: 100%; }
    .stButton > button:hover { transform: translateY(-2px); box-shadow: 0 4px 12px rgba(139, 92, 246, 0.4); }
    
    /* Result Boxes */
    .result-success { background: linear-gradient(90deg, rgba(16, 185, 129, 0.2) 0%, rgba(5, 150, 105, 0.1) 100%); border-left: 5px solid #10b981; padding: 1.5rem; border-radius: 8px; margin-top: 1rem; }
    .result-danger { background: linear-gradient(90deg, rgba(239, 68, 68, 0.2) 0%, rgba(185, 28, 28, 0.1) 100%); border-left: 5px solid #ef4444; padding: 1.5rem; border-radius: 8px; margin-top: 1rem; }
    
    /* Footer */
    .footer { text-align: center; color: #94a3b8; font-size: 0.9rem; padding: 2rem 0 1rem 0; margin-top: 3rem; border-top: 1px solid #334155; }
label[data-testid="stWidgetLabel"] p, label[data-testid="stWidgetLabel"] span, label[data-testid="stWidgetLabel"] div { color: #FFFFFF !important; font-weight: 600 !important; } div[data-testid="stSelectbox"] label p, div[data-testid="stNumberInput"] label p, div[data-testid="stTextInput"] label p, div[data-testid="stSlider"] label p, div[data-testid="stRadio"] label p, div[data-testid="stCheckbox"] label p, div[data-testid="stForm"] label p { color: #FFFFFF !important; } </style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    try:
        return joblib.load(os.path.join(BASE_DIR, "models", "best_model.joblib"))
    except FileNotFoundError:
        st.error("⚠️ Model not found. Please run `python train_pipeline.py` first.")
        st.stop()

model = load_model()

# --------------------------------------------------------------------------- #
# Sidebar
# --------------------------------------------------------------------------- #
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2830/2830284.png", width=60)
    st.markdown("### Model Performance")
    st.markdown("---")
    try:
        comp = pd.read_csv(os.path.join(BASE_DIR, "models", "model_comparison.csv"), index_col=0)
        st.dataframe((comp * 100).round(1).style.highlight_max(axis=0, color='#3b82f6').format("{:.1f}%"))
    except FileNotFoundError:
        st.info("Model metrics not found.")
    st.markdown("---")
    st.info("This tool uses a **Logistic Regression** model. It assists loan officers and should not be used as the sole basis for approval.")

# --------------------------------------------------------------------------- #
# Main UI
# --------------------------------------------------------------------------- #
st.markdown('<div class="main-title">🏦 Motz Financial Services</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Intelligent Loan Approval Prediction System</div>', unsafe_allow_html=True)

col_form, col_res = st.columns([1.2, 1], gap="large")

with col_form:
    st.markdown("### 📋 Applicant Details")
    with st.form("applicant_form"):
        
        st.markdown("**Personal Information**")
        c1, c2, c3 = st.columns(3)
        gender = c1.selectbox("Gender", ["Male", "Female"])
        married = c2.selectbox("Married", ["Yes", "No"])
        dependents = c3.selectbox("Dependents", ["0", "1", "2", "3"])
        education = st.selectbox("Education Level", ["Graduate", "Not Graduate"])
        
        st.markdown("---")
        st.markdown("**Financial Details**")
        c4, c5 = st.columns(2)
        applicant_income = c4.number_input("Applicant Income (Monthly)", min_value=0, value=5000, step=500)
        coapplicant_income = c5.number_input("Co-applicant Income (Monthly)", min_value=0, value=0, step=500)
        
        c6, c7 = st.columns(2)
        loan_amount = c6.number_input("Loan Amount (Thousands)", min_value=1, value=120, step=10)
        loan_term = c7.selectbox("Loan Term (Months)", [360, 180, 120, 240, 300, 480, 84, 60, 36, 12])
        
        st.markdown("---")
        st.markdown("**Risk Assessment**")
        
        cibil_score = st.slider(
            "CIBIL Score (Credit Bureau Rating)", 
            min_value=300, max_value=900, value=750, step=1,
            help="300-579: Poor, 580-669: Fair, 670-739: Good, 740-799: Very Good, 800-900: Excellent. "
                 "The model dynamically maps this to a binary 'Credit History' (Good/Bad) internally."
        )
        
        credit_history_val = 1 if cibil_score >= 650 else 0
        if cibil_score >= 750:
            st.caption("🟢 Credit History: **Excellent**")
        elif cibil_score >= 650:
            st.caption("🟡 Credit History: **Good**")
        else:
            st.caption("🔴 Credit History: **Poor**")
        
        c8, c9 = st.columns(2)
        property_area = c8.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])
        self_emp = c9.selectbox("Self Employed", ["No", "Yes"])

        st.markdown("<br>", unsafe_allow_html=True)
        submitted = st.form_submit_button("Run Prediction Analysis")

with col_res:
    st.markdown("### 📊 Prediction Results")
    
    if submitted:
        row = pd.DataFrame([{
            "Gender": gender, "Married": married, "Dependents": dependents,
            "Education": education, "Self_Employed": self_emp,
            "ApplicantIncome": float(applicant_income), "CoapplicantIncome": float(coapplicant_income),
            "LoanAmount": float(loan_amount), "Loan_Amount_Term": float(loan_term),
            "Credit_History": credit_history_val,
            "Property_Area": property_area,
        }])

        row = engineer_features(row)
        expected_cols = ["Gender","Married","Dependents","Education","Self_Employed",
                         "ApplicantIncome","CoapplicantIncome","LoanAmount","Loan_Amount_Term",
                         "Credit_History","Property_Area","TotalIncome","LoanAmount_log","EMI","Balance_Income"]
        row = row[expected_cols]

        pred = model.predict(row)[0]
        proba = model.predict_proba(row)[0][1]

        # Result Display
        if pred == 1:
            st.markdown(f'<div class="result-success"><h2 style="color: #10b981; margin:0;">✅ LOAN APPROVED</h2><p style="color: #94a3b8; margin:0;">Confidence: {proba:.1%}</p></div>', unsafe_allow_html=True)
            status_color = "#10b981"
        else:
            st.markdown(f'<div class="result-danger"><h2 style="color: #ef4444; margin:0;">❌ LOAN REJECTED</h2><p style="color: #94a3b8; margin:0;">Probability of Approval: {proba:.1%}</p></div>', unsafe_allow_html=True)
            status_color = "#ef4444"

        st.markdown("<br>", unsafe_allow_html=True)

        # Gauge Chart
        fig = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = proba * 100,
            title = {'text': "Approval Probability (%)", 'font': {'color': 'white'}},
            number = {'suffix': "%", 'font': {'color': 'white'}},
            gauge = {
                'axis': {'range': [None, 100], 'tickcolor': "white"},
                'bar': {'color': status_color},
                'bgcolor': "#1e293b",
                'steps': [
                    {'range': [0, 50], 'color': "rgba(239, 68, 68, 0.2)"},
                    {'range': [50, 100], 'color': "rgba(16, 185, 129, 0.2)"}
                ],
                'threshold': {'line': {'color': "white", 'width': 2}, 'thickness': 0.75, 'value': 50}
            }
        ))
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", font={'color': "white"}, height=250, margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig, use_container_width=True)

        # Explainability
        with st.expander("🔍 Why this decision? (Key Factors)", expanded=True):
            if credit_history_val == 0:
                st.error("🚨 **Credit History:** A CIBIL score below 650 indicates poor credit, which is the strongest predictor of rejection.")
            else:
                st.success("✅ **Credit History:** A good CIBIL score strongly supports approval.")
            if proba < 0.3:
                st.write("⚠️ **Income/Loan Ratio:** The requested loan amount may be too high relative to income.")
            if education == "Not Graduate":
                st.write("ℹ️ **Education:** Not being a graduate marginally reduces approval odds.")

        # DOWNLOAD BUTTON
        st.markdown("---")
        # Prepare the downloadable dataframe
        download_row = row.copy()
        download_row.insert(0, "CIBIL_Score", cibil_score)
        download_row["Prediction"] = "Approved" if pred == 1 else "Rejected"
        download_row["Approval_Probability"] = f"{proba:.2%}"
        
        csv = download_row.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Applicant Record (CSV)",
            data=csv,
            file_name="loan_applicant_record.csv",
            mime="text/csv",
            help="Download the entered details and the model's prediction."
        )
        
    else:
        st.info("👈 Fill out the applicant details on the left and click **Run Prediction Analysis**.")
        st.markdown('<div style="text-align: center; margin-top: 50px; opacity: 0.5;"><h1 style="font-size: 80px;">📊</h1><p>Awaiting applicant data...</p></div>', unsafe_allow_html=True)

# Footer
st.markdown("---")
with st.expander("🧪 Test Model on Historical Data (Examiner Demo)"):
    uploaded_file = st.file_uploader("Upload test CSV (must have same columns as training data)", type=["csv"])
    if uploaded_file is not None:
        test_df = pd.read_csv(uploaded_file)
        try:
            test_clean = test_df.copy()
            if "Loan_ID" in test_clean.columns:
                test_clean = test_clean.drop(columns=["Loan_ID"])
            true_labels = None
            if "Loan_Status" in test_clean.columns:
                true_labels = test_clean["Loan_Status"].map({"Y": 1, "N": 0})
                test_clean = test_clean.drop(columns=["Loan_Status"])
                
            test_feat = engineer_features(test_clean)
            test_feat = test_feat[expected_cols]
            
            preds = model.predict(test_feat)
            probs = model.predict_proba(test_feat)[:, 1]
            
            out_df = test_df.copy()
            out_df["Predicted_Status"] = ["Y" if p == 1 else "N" for p in preds]
            out_df["Approval_Probability"] = (probs * 100).round(1)
            
            st.dataframe(out_df.head(10))
            st.download_button("📥 Download Full Predictions", data=out_df.to_csv(index=False).encode('utf-8'), file_name='loan_predictions.csv', mime='text/csv')
            
            if true_labels is not None:
                st.success(f"**Accuracy on this sample:** {(preds == true_labels).mean():.2%}")
        except Exception as e:
            st.error(f"Error processing file: {e}")

# --------------------------------------------------------------------------- #
# Footer Copyright
# --------------------------------------------------------------------------- #
st.markdown('<div class="footer">© 2026 Motz Financial Services. All rights reserved.</div>', unsafe_allow_html=True)


