import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
import shap
import matplotlib.pyplot as plt
import os

st.set_page_config(page_title="Credit Card Fraud Detection", layout="wide")
st.title("🛡️ Credit Card Fraud Detection Dashboard")
st.markdown("""
This app downloads the Kaggle Credit Card Fraud dataset automatically, trains a fraud detection model,
and highlights suspicious transactions.
""")

# --- Step 1: Download dataset from Kaggle ---
st.subheader("Dataset Setup")
DATA_PATH = "creditcard.csv"

if not os.path.exists(DATA_PATH):
    st.info("Downloading dataset from Kaggle...")
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi

        api = KaggleApi()
        api.authenticate()

        dataset_path = 'mlg-ulb/creditcardfraud'
        api.dataset_download_files(dataset_path, path='.', unzip=True)
        st.success("Dataset downloaded successfully!")
    except Exception as e:
        st.error("Error downloading dataset. Please place 'creditcard.csv' in the folder or set up Kaggle API credentials.")
        st.stop()
else:
    st.success("Dataset already exists locally.")

# --- Step 2: Load dataset ---
df = pd.read_csv(DATA_PATH)
st.write(f"Dataset loaded with {len(df)} transactions.")

# --- Step 3: Dataset overview ---
st.subheader("Dataset Overview")
st.write(df.head())
st.write("Fraud Class Distribution:")
st.bar_chart(df['Class'].value_counts())

# --- Step 4: Feature Engineering ---
df['hour'] = (df['Time'] // 3600) % 24
df['high_amount'] = df['Amount'] > 2000

# Features for model
feature_cols = [c for c in df.columns if c not in ['Class', 'Time']]
X = df[feature_cols]
y = df['Class']

# --- Step 5: Train-Test Split ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# --- Step 6: Train Model ---
model = RandomForestClassifier(
    n_estimators=200,
    class_weight='balanced',
    random_state=42
)
model.fit(X_train, y_train)

# --- Step 7: Predict fraud ---
y_prob = model.predict_proba(X)[:, 1]
df['fraud_probability'] = y_prob
df['fraud_flag'] = df['fraud_probability'] > 0.5

# --- Step 8: Model Evaluation ---
st.subheader("📊 Model Evaluation")
y_pred = model.predict(X)
report = classification_report(y, y_pred, output_dict=True)
st.write(pd.DataFrame(report).transpose())
st.write("ROC-AUC Score:", roc_auc_score(y, y_prob))

# --- Step 9: Display Transactions ---
st.subheader("📄 All Transactions with Fraud Probability")
st.dataframe(df[['Time', 'Amount', 'hour', 'high_amount', 'fraud_probability', 'fraud_flag']])

st.subheader("🚨 Flagged / Suspicious Transactions")
flagged = df[df['fraud_flag']]
st.write(f"Total flagged: {len(flagged)}")
st.dataframe(flagged[['Time', 'Amount', 'hour', 'high_amount', 'fraud_probability']])

# --- Step 10: SHAP Explainability ---
st.subheader("🔍 Feature Importance via SHAP")
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X)
fig, ax = plt.subplots(figsize=(10,5))
shap.summary_plot(shap_values[1], X, plot_type="bar", show=False)
st.pyplot(fig)

st.success("✅ Fraud Detection Dashboard Ready!")
