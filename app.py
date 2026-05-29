import streamlit as st
import pandas as pd
import joblib
import plotly.graph_objects as go
import plotly.express as px

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(
    page_title="AI Loan Risk Intelligence Platform",
    page_icon="🏦",
    layout="wide"
)

# -----------------------------
# LOAD CSS
# -----------------------------
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(
            f"<style>{f.read()}</style>",
            unsafe_allow_html=True
        )

local_css("assets/style.css")

# -----------------------------
# LOAD MODEL
# -----------------------------
@st.cache_resource
def load_model():

    try:

        model = joblib.load(
            "saved_models/best_loan_default_model.joblib"
        )

        return model

    except Exception as e:

        st.error(
            f"MODEL LOAD ERROR: {type(e).__name__}"
        )

        st.error(str(e))

        st.stop()

model = load_model()

# -----------------------------
# HEADER
# -----------------------------
st.markdown("""
<div style='text-align:center'>
<h1 style='color:#1565C0'>
🏦 AI Loan Risk Intelligence Platform
</h1>

<h4 style='color:gray'>
Real-Time Credit Risk Assessment Dashboard
</h4>
</div>
<hr>
""", unsafe_allow_html=True)

# -----------------------------
# SIDEBAR
# -----------------------------
st.sidebar.title("📋 Applicant Information")

with st.sidebar.expander(
    "👤 Personal Details",
    expanded=True
):
    Age = st.slider(
        "Age",
        18,
        90,
        30
    )

    MaritalStatus = st.selectbox(
        "Marital Status",
        ["Single", "Married", "Divorced"]
    )

    HasDependents = st.selectbox(
        "Has Dependents",
        ["Yes", "No"]
    )

with st.sidebar.expander(
    "💰 Financial Details",
    expanded=True
):
    Income = st.number_input(
        "Annual Income ($)",
        min_value=0,
        value=50000,
        step=1000
    )

    CreditScore = st.slider(
        "Credit Score",
        300,
        850,
        700
    )

    MonthsEmployed = st.slider(
        "Months Employed",
        0,
        360,
        60
    )

    NumCreditLines = st.slider(
        "Number of Credit Lines",
        0,
        10,
        2
    )

    HasMortgage = st.selectbox(
        "Has Mortgage",
        ["Yes", "No"]
    )

with st.sidebar.expander(
    "🏦 Loan Details",
    expanded=True
):
    LoanAmount = st.number_input(
        "Loan Amount ($)",
        min_value=0,
        value=100000,
        step=5000
    )

    InterestRate = st.slider(
        "Interest Rate (%)",
        0.0,
        30.0,
        7.5
    )

    LoanTerm = st.slider(
        "Loan Term",
        12,
        60,
        36
    )

    DTIRatio = st.slider(
        "Debt-To-Income Ratio",
        0.0,
        1.0,
        0.3
    )

    LoanPurpose = st.selectbox(
        "Loan Purpose",
        [
            "Business",
            "Home",
            "Auto",
            "Education",
            "Other"
        ]
    )

with st.sidebar.expander(
    "🎓 Employment Details",
    expanded=True
):
    Education = st.selectbox(
        "Education",
        [
            "High School",
            "Bachelor's",
            "Master's",
            "PhD"
        ]
    )

    EmploymentType = st.selectbox(
        "Employment Type",
        [
            "Full-time",
            "Part-time",
            "Self-employed",
            "Unemployed"
        ]
    )

    HasCoSigner = st.selectbox(
        "Has Co-Signer",
        ["Yes", "No"]
    )

# -----------------------------
# INPUT DATA
# -----------------------------
input_df = pd.DataFrame({
    "Age":[Age],
    "Income":[Income],
    "LoanAmount":[LoanAmount],
    "CreditScore":[CreditScore],
    "MonthsEmployed":[MonthsEmployed],
    "NumCreditLines":[NumCreditLines],
    "InterestRate":[InterestRate],
    "LoanTerm":[LoanTerm],
    "DTIRatio":[DTIRatio],
    "Education":[Education],
    "EmploymentType":[EmploymentType],
    "MaritalStatus":[MaritalStatus],
    "HasMortgage":[HasMortgage],
    "HasDependents":[HasDependents],
    "LoanPurpose":[LoanPurpose],
    "HasCoSigner":[HasCoSigner]
})

# -----------------------------
# TABS
# -----------------------------
tab1, tab2, tab3 = st.tabs([
    "🔍 Prediction",
    "👤 Applicant Profile",
    "📊 Analytics"
])

with tab1:

    predict = st.button(
        "🚀 Predict Loan Risk",
        use_container_width=True
    )

    if predict:

        prediction = model.predict(
            input_df
        )[0]

        probability = model.predict_proba(
            input_df
        )[0][1]

        if probability < 0.30:
            risk = "Low Risk"
        elif probability < 0.70:
            risk = "Medium Risk"
        else:
            risk = "High Risk"

        decision = (
            "Default"
            if prediction == 1
            else "No Default"
        )

        st.success(
            "Prediction Generated Successfully"
        )

        col1, col2, col3, col4 = st.columns(4)

        col1.metric(
            "Default Probability",
            f"{probability:.2%}"
        )

        col2.metric(
            "Risk Level",
            risk
        )

        col3.metric(
            "Decision",
            decision
        )

        col4.metric(
            "Credit Score",
            CreditScore
        )

        gauge = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=probability*100,
                title={
                    "text":"Default Risk Score"
                },
                gauge={
                    "axis":{
                        "range":[0,100]
                    },
                    "steps":[
                        {"range":[0,30]},
                        {"range":[30,70]},
                        {"range":[70,100]}
                    ]
                }
            )
        )

        st.plotly_chart(
            gauge,
            use_container_width=True
        )

        st.subheader(
            "🤖 AI Recommendation"
        )

        if probability < 0.30:

            st.success("""
            ✅ APPROVE LOAN

            Applicant demonstrates
            strong repayment capability.
            """)

        elif probability < 0.70:

            st.warning("""
            ⚠ MANUAL REVIEW REQUIRED

            Moderate default risk detected.
            """)

        else:

            st.error("""
            ❌ HIGH DEFAULT RISK

            Recommend rejection or
            request additional collateral.
            """)

        report = pd.DataFrame({
            "Prediction":[decision],
            "Risk":[risk],
            "Probability":[
                f"{probability:.2%}"
            ]
        })

        st.download_button(
            "📥 Download Report",
            report.to_csv(index=False),
            "loan_report.csv",
            "text/csv"
        )

with tab2:

    st.subheader(
        "Applicant Profile"
    )

    st.dataframe(
        input_df,
        use_container_width=True
    )

with tab3:

    st.subheader(
        "Financial Analysis"
    )

    ratio = LoanAmount / max(
        Income,
        1
    )

    st.metric(
        "Loan-To-Income Ratio",
        f"{ratio:.2f}"
    )

    pie = px.pie(
        values=[
            Income,
            LoanAmount
        ],
        names=[
            "Income",
            "Loan Amount"
        ],
        title="Income vs Loan Amount"
    )

    st.plotly_chart(
        pie,
        use_container_width=True
    )

# -----------------------------
# FOOTER
# -----------------------------
st.markdown("---")

st.caption("""
AI Loan Risk Prediction System

Built with:
Python • Streamlit • Scikit-Learn • Plotly
""")