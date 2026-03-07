import streamlit as st
import numpy as np
import pandas as pd
import pickle
import os
import joblib

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Employee Productivity Prediction",
    page_icon="📊",
    layout="centered",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:wght@300;400;500&display=swap');

*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: #0d0f14;
    color: #e8e6e0;
}

.stApp {
    background: #0d0f14;
}

/* Header */
.hero-title {
    font-family: 'Syne', sans-serif;
    font-size: 2.6rem;
    font-weight: 800;
    background: linear-gradient(135deg, #f5c842 0%, #f0a500 50%, #e07b00 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    letter-spacing: -1px;
    line-height: 1.1;
    margin-bottom: 0.3rem;
}

.hero-sub {
    font-family: 'DM Sans', sans-serif;
    font-size: 1rem;
    color: #7a7a8c;
    font-weight: 300;
    letter-spacing: 0.5px;
    margin-bottom: 2rem;
}

/* Toggle */
.toggle-container {
    display: flex;
    background: #1a1d26;
    border: 1px solid #2a2d3a;
    border-radius: 14px;
    padding: 5px;
    gap: 4px;
    width: fit-content;
    margin-bottom: 2rem;
}

.toggle-btn {
    padding: 10px 28px;
    border-radius: 10px;
    font-family: 'Syne', sans-serif;
    font-weight: 600;
    font-size: 0.9rem;
    cursor: pointer;
    border: none;
    transition: all 0.25s ease;
}

.toggle-active {
    background: linear-gradient(135deg, #f5c842, #e07b00);
    color: #0d0f14;
    box-shadow: 0 4px 15px rgba(245, 200, 66, 0.3);
}

.toggle-inactive {
    background: transparent;
    color: #7a7a8c;
}

/* Section card */
.section-card {
    background: #13161f;
    border: 1px solid #22263a;
    border-radius: 20px;
    padding: 2rem;
    margin-bottom: 1.5rem;
}

.section-label {
    font-family: 'Syne', sans-serif;
    font-size: 0.75rem;
    font-weight: 700;
    letter-spacing: 2.5px;
    text-transform: uppercase;
    color: #f5c842;
    margin-bottom: 1.2rem;
}

/* Streamlit widget overrides */
.stSelectbox label, .stNumberInput label, .stMultiSelect label {
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.85rem !important;
    color: #9a9ab0 !important;
    font-weight: 500 !important;
    letter-spacing: 0.3px !important;
}

.stSelectbox > div > div,
.stNumberInput > div > div > input,
.stMultiSelect > div > div {
    background: #1a1d26 !important;
    border: 1px solid #2a2d3a !important;
    border-radius: 10px !important;
    color: #e8e6e0 !important;
    font-family: 'DM Sans', sans-serif !important;
}

.stSelectbox > div > div:focus-within,
.stNumberInput > div > div:focus-within,
.stMultiSelect > div > div:focus-within {
    border-color: #f5c842 !important;
    box-shadow: 0 0 0 2px rgba(245, 200, 66, 0.15) !important;
}

/* Predict button */
.stButton > button {
    background: linear-gradient(135deg, #f5c842 0%, #e07b00 100%) !important;
    color: #0d0f14 !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 700 !important;
    font-size: 1rem !important;
    letter-spacing: 0.5px !important;
    padding: 0.75rem 2.5rem !important;
    border: none !important;
    border-radius: 12px !important;
    width: 100% !important;
    transition: all 0.2s ease !important;
    box-shadow: 0 4px 20px rgba(245, 200, 66, 0.2) !important;
    margin-top: 1rem !important;
}

.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 30px rgba(245, 200, 66, 0.35) !important;
}

/* Result box */
.result-box {
    background: linear-gradient(135deg, rgba(245,200,66,0.08), rgba(224,123,0,0.08));
    border: 1px solid rgba(245, 200, 66, 0.3);
    border-radius: 16px;
    padding: 1.8rem 2rem;
    text-align: center;
    margin-top: 1.5rem;
}

.result-label {
    font-family: 'DM Sans', sans-serif;
    font-size: 0.8rem;
    color: #9a9ab0;
    letter-spacing: 2px;
    text-transform: uppercase;
    margin-bottom: 0.5rem;
}

.result-value {
    font-family: 'Syne', sans-serif;
    font-size: 2.8rem;
    font-weight: 800;
    background: linear-gradient(135deg, #f5c842, #e07b00);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}

.result-sub {
    font-size: 0.85rem;
    color: #7a7a8c;
    margin-top: 0.5rem;
}

/* Divider */
.divider {
    height: 1px;
    background: linear-gradient(90deg, transparent, #2a2d3a, transparent);
    margin: 1.5rem 0;
}

/* Warning / info */
.stAlert {
    background: #1a1d26 !important;
    border: 1px solid #2a2d3a !important;
    border-radius: 10px !important;
    color: #e8e6e0 !important;
}

/* Hide Streamlit branding */
#MainMenu, footer, header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ── Encoders / Data ───────────────────────────────────────────────────────────

DESIGNATION_ORDER = ['Junior', 'Executive', 'Lead', 'Senior', 'Manager']
EDUCATION_ORDER   = ['High School', 'Bachelor', 'Master', 'PhD']
DEPARTMENTS       = ['Finance', 'HR', 'IT', 'Marketing', 'Operations', 'Sales']
WORK_LOCATIONS    = ['Bangalore', 'Chennai', 'Delhi', 'Hyderabad', 'Kolkata', 'Mumbai', 'Noida', 'Pune']
SKILLS_LIST       = ['Accounting','Analytics','C++','CRM','Cloud Computing','Communication',
                     'Conflict Management','Content Creation','Data Analysis','Excel','Finance',
                     'Java','Logistics','Marketing','Negotiation','Project Management',
                     'Python','Recruitment','SEO','SQL']

DESIGNATION_MAP   = {0:'Junior', 1:'Executive', 2:'Lead', 3:'Senior', 4:'Manager'}

def encode_salary_features(age, gender, designation, exp, prod_score,
                            edu_level, perf_rating, last_promo_year,
                            work_location, department, skillset):
    gender_val = 1 if gender == 'Male' else 0
    des_val    = DESIGNATION_ORDER.index(designation)
    edu_val    = EDUCATION_ORDER.index(edu_level)

    # OneHotEncode WorkLocation (drop first = Bangalore)
    all_wl = WORK_LOCATIONS[1:]  # Chennai, Delhi, Hyderabad, Kolkata, Mumbai, Noida, Pune
    wl_enc = [1 if w == work_location else 0 for w in all_wl]

    # OneHotEncode Department (drop first = Finance)
    all_dep = DEPARTMENTS[1:]    # HR, IT, Marketing, Operations, Sales
    dep_enc = [1 if d == department else 0 for d in all_dep]

    # MultiLabelBinarize skills
    skill_enc = [1 if s in skillset else 0 for s in SKILLS_LIST]

    features = np.array([age, gender_val, des_val, exp, prod_score,
                         edu_val, perf_rating, last_promo_year]
                        + wl_enc + dep_enc + skill_enc, dtype=float).reshape(1, -1)
    return features   # shape (1, 40)


def predict_salary(features, lasso_model, poly_features=None):
    """Use Lasso model directly"""
    return lasso_model.predict(features)[0]


def predict_designation(salary, exp, prod_score, svm_model):
    arr = np.array([[salary, exp, prod_score]], dtype=float)
    pred = svm_model.predict(arr)[0]
    return DESIGNATION_MAP.get(int(pred), str(pred))


# ── Attempt to load saved models ──────────────────────────────────────────────
@st.cache_resource
def load_models():
    models = {}
    for name, path in [('lasso', 'lasso_model.pkl'),
                        ('svm',   'svm_model.pkl')]:
        if os.path.exists(path):
            models[name] = joblib.load(path)   # ← joblib not pickle
        else:
            models[name] = None
    return models

models = load_models()

# ── UI ────────────────────────────────────────────────────────────────────────

st.markdown('<div class="hero-title">Employee Productivity<br>Prediction</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-sub">ML-powered salary & designation forecasting</div>', unsafe_allow_html=True)

# Toggle via radio (styled as buttons)
mode = st.radio(
    "",
    ["💰  Salary Prediction", "🏷️  Designation Prediction"],
    horizontal=True,
    label_visibility="collapsed",
)

st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
#  SALARY PREDICTION
# ══════════════════════════════════════════════════════════════════════════════
if "Salary" in mode:
    st.markdown('<div class="section-label">Personal Details</div>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        age = st.number_input("Age", min_value=18, max_value=65, value=25, step=1)
    with col2:
        gender = st.selectbox("Gender", ["Male", "Female"])
    with col3:
        exp = st.number_input("Experience (Years)", min_value=0, max_value=40, value=2, step=1)

    col4, col5 = st.columns(2)
    with col4:
        designation = st.selectbox("Designation", DESIGNATION_ORDER)
    with col5:
        edu_level = st.selectbox("Education Level", EDUCATION_ORDER)

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    st.markdown('<div class="section-label">Performance Metrics</div>', unsafe_allow_html=True)

    col6, col7, col8 = st.columns(3)
    with col6:
        prod_score = st.number_input("Productivity Score", min_value=1, max_value=100, value=65, step=1)
    with col7:
        perf_rating = st.number_input("Performance Rating", min_value=1, max_value=5, value=4, step=1)
    with col8:
        last_promo = st.number_input("Last Promotion Year", min_value=2000, max_value=2025, value=2022, step=1)

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    st.markdown('<div class="section-label">Work Details</div>', unsafe_allow_html=True)

    col9, col10 = st.columns(2)
    with col9:
        work_loc = st.selectbox("Work Location", sorted(WORK_LOCATIONS))
    with col10:
        department = st.selectbox("Department", sorted(DEPARTMENTS))

    skillset = st.multiselect("Skillset", SKILLS_LIST, default=["Python", "Data Analysis"])

    if st.button("Predict Salary →"):
        if models['lasso'] is None:
            # Fallback: rule-based estimate when model not uploaded
            base = 25000
            base += exp * 2500
            base += DESIGNATION_ORDER.index(designation) * 8000
            base += EDUCATION_ORDER.index(edu_level) * 4000
            base += (prod_score - 50) * 200
            base += (perf_rating - 3) * 3000
            dept_bonus = {'IT': 8000, 'Finance': 6000, 'Marketing': 4000,
                          'Sales': 3000, 'HR': 2000, 'Operations': 2500}
            base += dept_bonus.get(department, 0)
            base += len(skillset) * 500
            predicted = max(25000, base)
            note = "*(Estimated — upload model files for ML prediction)*"
        else:
            feats = encode_salary_features(age, gender, designation, exp, prod_score,
                                           edu_level, perf_rating, last_promo,
                                           work_loc, department, skillset)
            predicted = predict_salary(feats, models['lasso'])
            note = "*(Lasso Regression model)*"

        st.markdown(f"""
        <div class="result-box">
            <div class="result-label">Predicted Annual Salary</div>
            <div class="result-value">₹{predicted:,.0f}</div>
            <div class="result-sub">{note}</div>
        </div>
        """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
#  DESIGNATION PREDICTION
# ══════════════════════════════════════════════════════════════════════════════
else:
    st.markdown('<div class="section-label">Employee Profile</div>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        salary = st.number_input("Current Salary (₹)", min_value=20000, max_value=200000,
                                  value=60000, step=1000)
    with col2:
        exp = st.number_input("Experience (Years)", min_value=0, max_value=40, value=5, step=1)
    with col3:
        prod_score = st.number_input("Productivity Score", min_value=1, max_value=100, value=75, step=1)

    st.markdown("""
    <div style="background:#1a1d26;border:1px solid #2a2d3a;border-radius:12px;padding:1rem 1.2rem;margin-top:1rem;">
        <div style="font-size:0.78rem;color:#7a7a8c;font-family:'DM Sans',sans-serif;">
            <b style="color:#f5c842;">Model:</b> SVM (RBF Kernel) &nbsp;|&nbsp;
            <b style="color:#f5c842;">Features:</b> Salary, Experience Years, Productivity Score &nbsp;|&nbsp;
            <b style="color:#f5c842;">Classes:</b> Junior → Executive → Lead → Senior → Manager
        </div>
    </div>
    """, unsafe_allow_html=True)

    if st.button("Predict Designation →"):
        if models['svm'] is None:
            # Fallback rule-based
            score = 0
            score += min(exp / 5, 4)
            score += min(salary / 30000, 4)
            score += (prod_score - 50) / 25
            idx = max(0, min(4, int(score)))
            predicted_desig = DESIGNATION_MAP[idx]
            note = "*(Estimated — upload model files for ML prediction)*"
        else:
            predicted_desig = predict_designation(salary, exp, prod_score, models['svm'])
            note = "*(SVM — RBF Kernel)*"

        desig_desc = {
            'Junior':    'Entry-level role suited for early-career employees.',
            'Executive': 'Mid-level contributor managing independent tasks.',
            'Lead':      'Technical lead guiding a small team.',
            'Senior':    'Senior individual contributor with domain expertise.',
            'Manager':   'People manager overseeing teams and strategy.'
        }

        st.markdown(f"""
        <div class="result-box">
            <div class="result-label">Predicted Designation</div>
            <div class="result-value">{predicted_desig}</div>
            <div class="result-sub">{desig_desc.get(predicted_desig, '')} &nbsp;{note}</div>
        </div>
        """, unsafe_allow_html=True)

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="text-align:center;margin-top:3rem;padding-top:1.5rem;
border-top:1px solid #1a1d26;font-size:0.75rem;color:#3a3d4a;
font-family:'DM Sans',sans-serif;">
    Employee Productivity Analysis · ML Mini Project
</div>
""", unsafe_allow_html=True)
