import os
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from fpdf import FPDF


st.set_page_config(page_title="Parkinson's Prediction", layout='wide', initial_sidebar_state='expanded')

BASE_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'model.pkl')
DATA_PATH = os.path.join(BASE_DIR, 'dataset', 'Parkinsson disease.csv')


def create_pdf_report(entry, model_info, out_path):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font('Arial', 'B', 16)
    pdf.cell(0, 10, "Parkinson's Prediction Report", ln=True)
    pdf.ln(4)
    pdf.set_font('Arial', '', 12)
    pdf.cell(0, 8, f"Model: {model_info['model']}", ln=True)
    pdf.cell(0, 8, f"Dataset: {model_info.get('dataset', '')}", ln=True)
    pdf.cell(0, 8, f"Accuracy: {model_info['accuracy']*100:.2f}%", ln=True)
    pdf.ln(4)
    pdf.cell(0, 8, 'Prediction: ' + ("Parkinson's Disease Detected" if entry['prediction'] == 1 else 'Healthy'), ln=True)
    if entry.get('proba'):
        pdf.cell(0, 8, f"Confidence: {max(entry['proba'])*100:.2f}%", ln=True)
    pdf.cell(0, 8, f"Timestamp: {entry.get('timestamp', '')}", ln=True)
    pdf.ln(6)
    pdf.cell(0, 8, 'Input features:', ln=True)
    pdf.set_font('Courier', '', 10)
    for k, v in entry['features'].items():
        pdf.multi_cell(0, 5, f" - {k}: {v}")
    pdf.output(out_path)


if not os.path.exists(MODEL_PATH):
    st.error('Model not found. Run `python train_model.py` to create `models/model.pkl`.')
    st.stop()


model = joblib.load(MODEL_PATH)

# load dataset
if not os.path.exists(DATA_PATH):
    st.error('Dataset not found at dataset/Parkinsson disease.csv')
    st.stop()

df = pd.read_csv(DATA_PATH)
df = df.drop(columns=['name'], errors='ignore')

# group features
FEATURE_GROUPS = {
    'Frequency Features': ['MDVP:Fo(Hz)', 'MDVP:Fhi(Hz)', 'MDVP:Flo(Hz)'],
    'Jitter Features': ['MDVP:Jitter(%)', 'MDVP:Jitter(Abs)', 'MDVP:RAP', 'MDVP:PPQ', 'Jitter:DDP'],
    'Shimmer Features': ['MDVP:Shimmer', 'MDVP:Shimmer(dB)', 'Shimmer:APQ3', 'Shimmer:APQ5', 'MDVP:APQ', 'Shimmer:DDA'],
    'Noise Features': ['NHR', 'HNR'],
    'Nonlinear Features': ['RPDE', 'DFA', 'spread1', 'spread2', 'D2', 'PPE']
}

ALL_FEATURES = [f for grp in FEATURE_GROUPS.values() for f in grp]


def key_for(name: str) -> str:
    return name.replace(' ', '_').replace(':', '_').replace('(', '').replace(')', '').replace('%', 'pct').replace('.', '_')


def pretty_label(pred: int) -> str:
    return "Parkinson's Disease Detected" if int(pred) == 1 else 'Healthy'


def confidence_pct(proba: np.ndarray) -> float:
    return float(proba.max()) * 100.0


def compute_model_info():
    X = df.drop(columns=['status'])
    y = df['status']
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=44)
    y_pred = model.predict(x_test)
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    model_name = None
    try:
        if hasattr(model, 'named_steps'):
            model_name = type(list(model.named_steps.values())[-1]).__name__
        else:
            model_name = type(model).__name__
    except Exception:
        model_name = type(model).__name__
    return {
        'model': model_name,
        'accuracy': acc,
        'dataset': os.path.basename(DATA_PATH),
        'n_features': X.shape[1],
        'n_records': X.shape[0],
        'confusion_matrix': cm
    }


model_info = compute_model_info()


# styles and header
st.markdown(
    """
<style>
body { background-color: #f7fbfc; }
.card {padding:16px;background:#ffffff;border-radius:10px;box-shadow:0 6px 18px rgba(0,0,0,0.06);}
.green {color:#0f5132;background:#e6ffed;padding:8px;border-radius:6px}
.red {color:#7f1d1d;background:#ffe6e6;padding:8px;border-radius:6px}
.muted {color:#6c757d}
.metric {font-size:20px;font-weight:700}
.small {font-size:12px;color:#6c757d}
.footer {color:#6c757d;font-size:12px;margin-top:20px}
@media (max-width: 600px) {.card{padding:10px}}
</style>
""",
    unsafe_allow_html=True,
)

st.title("Parkinson's Disease Prediction — Medical AI")
st.markdown("Modern, responsive demo for a portfolio — includes model info, predictions, and exports.")


# Sidebar: About and tech info
with st.sidebar:
    st.header('Project Information')
    st.write('Parkinson\'s Disease Prediction System')
    st.markdown(f"- **Dataset:** UCI Parkinson's Dataset")
    st.markdown(f"- **Records:** {model_info['n_records']}")
    st.markdown(f"- **Features:** {model_info['n_features']}")
    st.markdown(f"- **Model:** {model_info['model']}")
    st.markdown(f"- **Accuracy:** {model_info['accuracy']*100:.2f}%")
    st.markdown('---')
    st.subheader('Instructions')
    st.write('Enter feature values (placeholders are dataset means), or load a sample patient.')
    st.markdown('---')
    # Top feature importance will be rendered below
    st.subheader('Top features')
    try:
        if hasattr(model, 'named_steps'):
            estimator = list(model.named_steps.values())[-1]
        else:
            estimator = model
        if hasattr(estimator, 'coef_'):
            coefs = np.abs(estimator.coef_).flatten()
            feat_names = ALL_FEATURES[:len(coefs)]
            imp = pd.Series(coefs, index=feat_names).sort_values(ascending=False)[:10]
            st.bar_chart(imp)
        else:
            st.write('Importance not available')
    except Exception:
        st.write('Error computing importance')


# Main layout
col_main, col_right = st.columns([3, 1])

with col_right:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('**Model Summary**')
    st.write(f"**Model:** {model_info['model']}")
    st.write(f"**Accuracy:** {model_info['accuracy']*100:.2f}%")
    st.write(f"**Records:** {model_info['n_records']}")
    st.write(f"**Features:** {model_info['n_features']}")
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="card" style="margin-top:12px">', unsafe_allow_html=True)
    st.markdown('**Confusion matrix**')
    fig, ax = plt.subplots(figsize=(3.5, 2.6))
    sns.heatmap(model_info['confusion_matrix'], annot=True, fmt='d', cmap='Blues', ax=ax)
    ax.set_xlabel('Predicted'); ax.set_ylabel('Actual')
    st.pyplot(fig)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="card" style="margin-top:12px">', unsafe_allow_html=True)
    st.markdown('**PCA (2 components)**')
    try:
        X = df.drop(columns=['status'])
        pca = PCA(n_components=2)
        proj = pca.fit_transform(X)
        fig2 = go.Figure(data=go.Scatter(x=proj[:, 0], y=proj[:, 1], mode='markers', marker=dict(color=df['status'], colorscale='Turbo'), text=[f"Idx {i}" for i in df.index]))
        fig2.update_layout(height=300, margin=dict(t=10, b=10))
        st.plotly_chart(fig2, use_container_width=True)
    except Exception:
        st.write('PCA visualization error')
    st.markdown('</div>', unsafe_allow_html=True)


with col_main:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader('Patient Inputs')
    st.info("These biomedical voice features are taken from the Parkinson's Disease dataset and are used by the machine learning model to make predictions.")
    st.write('Group sections below; placeholders are dataset means.')

    # Sample selector and buttons
    sample_idx = st.selectbox('Sample patients (index)', options=list(range(len(df))), format_func=lambda i: f'Row {i} (status={int(df.loc[i, "status"])})')
    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button('Load Sample Patient'):
            sample = df.loc[sample_idx]
            for f in ALL_FEATURES:
                st.session_state[key_for(f)] = float(sample[f])
            st.success(f'Loaded sample {sample_idx}')
    with col2:
        if st.button('Reset Form'):
            for f in ALL_FEATURES:
                st.session_state[key_for(f)] = float(df[f].mean())
            st.info('Form reset to dataset means')

    # form
    form = st.form('predict_form')
    inputs = []
    FEATURE_DESCRIPTIONS = {
        'MDVP:Fo(Hz)': 'Average voice frequency (Hz).',
        'MDVP:Fhi(Hz)': 'Maximum fundamental frequency (Hz).',
        'MDVP:Flo(Hz)': 'Minimum fundamental frequency (Hz).',
        'MDVP:Jitter(%)': 'Relative jitter (%). Frequency variation.',
        'MDVP:Jitter(Abs)': 'Absolute jitter. Frequency variation.',
        'MDVP:RAP': 'Relative average perturbation.',
        'MDVP:PPQ': 'Pitch period perturbation quotient.',
        'Jitter:DDP': 'Difference of differences of pitch periods.',
        'MDVP:Shimmer': 'Shimmer measure (amplitude variation).',
        'MDVP:Shimmer(dB)': 'Shimmer in dB.',
        'Shimmer:APQ3': 'Amplitude perturbation quotient (3).',
        'Shimmer:APQ5': 'Amplitude perturbation quotient (5).',
        'MDVP:APQ': 'Amplitude perturbation quotient.',
        'Shimmer:DDA': 'Derivative of amplitude variation.',
        'NHR': 'Noise-to-Harmonic Ratio.',
        'HNR': 'Harmonics-to-Noise Ratio.',
        'RPDE': 'Recurrence period density entropy (nonlinear).',
        'DFA': 'Detrended fluctuation analysis (nonlinear).',
        'spread1': 'Nonlinear spread measure 1.',
        'spread2': 'Nonlinear spread measure 2.',
        'D2': 'Correlation dimension estimate.',
        'PPE': 'Pitch period entropy.'
    }

    for group_name, feats in FEATURE_GROUPS.items():
        with form.expander(group_name, expanded=True):
            cols = st.columns(3)
            for i, feat in enumerate(feats):
                k = key_for(feat)
                default = float(df[feat].mean())
                help_txt = FEATURE_DESCRIPTIONS.get(feat, '')
                val = cols[i % 3].number_input(label=feat, value=st.session_state.get(k, default), key=k, format='%.6f', help=help_txt)
                inputs.append((feat, k))
            cols[0].caption('Tip: ' + group_name)

    submitted = form.form_submit_button('Predict')
    st.markdown('</div>', unsafe_allow_html=True)

    if submitted:
        # validate
        missing = [feat for feat, k in inputs if (str(st.session_state.get(k, '')).strip() == '')]
        if missing:
            st.error('Please fill all fields. Missing: ' + ', '.join(missing))
        else:
            try:
                vals = [float(st.session_state[k]) for _, k in inputs]
                arr = np.array(vals).reshape(1, -1)
                pred = model.predict(arr)[0]
                proba = model.predict_proba(arr)[0] if hasattr(model, 'predict_proba') else None

                # history with timestamp and confidence
                from datetime import datetime
                hist = st.session_state.setdefault('history', [])
                entry = {
                    'features': {f: float(st.session_state[k]) for f, k in inputs},
                    'prediction': int(pred),
                    'proba': proba.tolist() if proba is not None else None,
                    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                }
                hist.insert(0, entry)
                st.session_state['history'] = hist[:50]

                # result card
                if pred == 1:
                    st.markdown('<div class="red"><h3>🚨 Parkinson\'s Disease Detected</h3></div>', unsafe_allow_html=True)
                else:
                    st.markdown('<div class="green"><h3>✅ Healthy</h3></div>', unsafe_allow_html=True)

                # confidence and recommendation
                if proba is not None:
                    pct = confidence_pct(proba)
                    st.markdown(f"<div class='card' style='margin-top:8px'><strong>Confidence:</strong> <span class='metric'>{pct:.2f}%</span></div>", unsafe_allow_html=True)
                    # recommendation
                    rec = 'Recommend clinical follow-up with a neurologist and confirmatory tests.' if pred == 1 else 'No immediate action recommended; continue routine monitoring.'
                    st.markdown(f"<div class='card' style='margin-top:8px'><strong>Recommendation:</strong> <div class='small'>{rec}</div></div>", unsafe_allow_html=True)
                    # probability bar
                    figp = go.Figure(go.Bar(x=[f'Class {i}' for i in range(len(proba))], y=proba, marker_color=['#2ecc71' if i == 0 else '#e74c3c' for i in range(len(proba))]))
                    figp.update_layout(title='Class probabilities', yaxis_tickformat='.2f', height=300)
                    st.plotly_chart(figp, use_container_width=True)
                    # gauge
                    gauge = go.Figure(go.Indicator(mode='gauge+number', value=pct, number={'suffix': '%'}, gauge={'axis': {'range': [0, 100]}}))
                    st.plotly_chart(gauge, use_container_width=True)
                else:
                    st.write('Confidence not available for this model')

            except Exception as e:
                st.error('Prediction error: ' + str(e))

    # prediction history and export
    st.markdown('---')
    st.subheader('Session prediction history')
    hist = st.session_state.get('history', [])
    if hist:
        clear_col, download_col = st.columns([1, 1])
        with clear_col:
            if st.button('Clear History'):
                st.session_state['history'] = []
                st.success('History cleared')
        with download_col:
            if st.button('Download latest report as PDF'):
                latest = hist[0]
                out_pdf = os.path.join(BASE_DIR, 'prediction_report.pdf')
                create_pdf_report(latest, model_info, out_pdf)
                with open(out_pdf, 'rb') as f:
                    st.download_button('Download PDF', data=f, file_name='prediction_report.pdf', mime='application/pdf')

    for i, h in enumerate(hist):
        ts = h.get('timestamp', '')
        conf = f"{max(h['proba'])*100:.2f}%" if h.get('proba') else 'N/A'
        st.markdown(f"**{i+1}.** {pretty_label(h['prediction'])} — Confidence: {conf} — {ts}")
        with st.expander('View features', expanded=False):
            for fk, fv in h['features'].items():
                st.write(f"- {fk}: {fv}")


    st.markdown('---')
    st.markdown('**Project Workflow**')
    st.markdown('Dataset → Data Preprocessing → Logistic Regression → Prediction → Confidence Score → PDF Report')
    st.markdown('---')
    st.markdown('<div class="footer">Disclaimer: This project is intended for educational and research purposes only and should not be used for medical diagnosis.</div>', unsafe_allow_html=True)
