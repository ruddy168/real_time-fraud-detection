import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from datetime import datetime, timedelta
import random

# Page configuration
st.set_page_config(page_title="Real-Time Fraud Detection", layout="wide")
st.title("🛡️ Real-Time Fraud Detection System")
st.markdown("Upload a CSV or simulate transactions and detect fraud in real-time.")

# User input mode
option = st.radio("Choose Input Mode", ["Upload CSV", "Simulate Transactions"])
df = None  # Initialize

# CSV Upload mode
if option == "Upload CSV":
    uploaded_file = st.file_uploader("Upload transaction CSV", type=["csv"])
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.write("Uploaded columns:", df.columns.tolist())

# Simulate mode
elif option == "Simulate Transactions":
    from faker import Faker
    fake = Faker()
    def generate_transaction():
        return {
            "transaction_id": fake.uuid4(),
            "amount": round(random.uniform(10, 10000), 2),
            "location": random.choice(["USA", "India", "UK", "Russia", "Nigeria"]),
            "timestamp": datetime.now() - timedelta(minutes=random.randint(0, 1440)),
            "merchant_type": random.choice(["Retail", "Online", "ATM", "Crypto", "Travel"]),
        }
    df = pd.DataFrame([generate_transaction() for _ in range(50)])
    st.write("Simulated columns:", df.columns.tolist())

# Proceed if data is ready
if df is not None:
    # Convert timestamp safely
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
    df['hour'] = df['timestamp'].dt.hour.fillna(-1).astype(int)

    # Feature Engineering
    df['amount_over_5000'] = df['amount'] > 5000
    df['odd_hour'] = df['hour'].isin([0, 1, 2, 3, 4])
    df['suspicious_location'] = df['location'].isin(["Russia", "Nigeria", "North Korea"])
    df['crypto_use'] = df['merchant_type'] == "Crypto"

    # Model input
    features = ['amount_over_5000', 'odd_hour', 'suspicious_location', 'crypto_use']
    X = df[features]

    # Simple fraud label (for demo) — 1 if any rule matches
    y = [1 if any([a, b, c, d]) else 0 for a, b, c, d in zip(
        X['amount_over_5000'], X['odd_hour'], X['suspicious_location'], X['crypto_use']
    )]

    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)

    # Predict fraud
    df['fraud_probability'] = model.predict_proba(X)[:, 1]
    df['fraud_flag'] = df['fraud_probability'] > 0.5

    # Show full table
    st.subheader("📄 Transaction Data")
    st.dataframe(df[['transaction_id', 'amount', 'location', 'merchant_type', 'fraud_probability', 'fraud_flag']])

    # Show suspicious
    st.subheader("🚨 Suspicious Transactions")
    flagged = df[df['fraud_flag']]
    st.write(f"Total flagged: {len(flagged)}")
    st.dataframe(flagged[['transaction_id', 'amount', 'location', 'merchant_type', 'fraud_probability']])
else:
    st.warning("Please upload a valid CSV or simulate transactions to proceed.")