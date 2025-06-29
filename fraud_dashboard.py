import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from datetime import datetime
import random

st.set_page_config(page_title="Real-Time Fraud Detection", layout="wide")

st.title("🛡️ Real-Time Fraud Detection System")
st.markdown("Upload a CSV or simulate transactions and detect fraud in real-time.")

# Upload or simulate
option = st.radio("Choose Input Mode", ["Upload CSV", "Simulate Transactions"])

if option == "Upload CSV":
    uploaded_file = st.file_uploader("Upload transaction CSV", type=["csv"])
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
else:
    from faker import Faker
    fake = Faker()
    def generate_transaction():
        return {
            "transaction_id": fake.uuid4(),
            "amount": round(random.uniform(10, 10000), 2),
            "location": random.choice(["USA", "India", "UK", "Russia", "Nigeria"]),
            "timestamp": datetime.now(),
            "merchant_type": random.choice(["Retail", "Online", "ATM", "Crypto", "Travel"]),
        }
    df = pd.DataFrame([generate_transaction() for _ in range(50)])

# Feature Engineering
df['hour'] = pd.to_datetime(df['timestamp']).dt.hour
df['amount_over_5000'] = df['amount'] > 5000
df['odd_hour'] = df['hour'].isin([0,1,2,3,4])
df['suspicious_location'] = df['location'].isin(["Russia", "Nigeria", "North Korea"])
df['crypto_use'] = df['merchant_type'] == "Crypto"

# Model (simplified)
features = ['amount_over_5000', 'odd_hour', 'suspicious_location', 'crypto_use']
X = df[features]
model = RandomForestClassifier(n_estimators=100, random_state=42)
y = [1 if any([a,b,c,d]) else 0 for a,b,c,d in zip(X['amount_over_5000'], X['odd_hour'], X['suspicious_location'], X['crypto_use'])]
model.fit(X, y)

df['fraud_probability'] = model.predict_proba(X)[:,1]
df['fraud_flag'] = df['fraud_probability'] > 0.5

# Display
st.subheader("📄 Transaction Data")
st.dataframe(df[['transaction_id', 'amount', 'location', 'merchant_type', 'fraud_probability', 'fraud_flag']])

st.subheader("🚨 Suspicious Transactions")
suspicious = df[df['fraud_flag']]
st.write(f"Total flagged: {len(suspicious)}")
st.dataframe(suspicious[['transaction_id', 'amount', 'location', 'merchant_type', 'fraud_probability']])
