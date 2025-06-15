import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder

FILE = "financial_fraud_detection_dataset.csv"   # <-- nowa nazwa pliku
OUTPUT = "processed_data.pkl"
SAMPLE = 500_000  # Zmień w zależności od RAM

df = pd.read_csv(FILE, nrows=SAMPLE)

# Feature engineering
df['timestamp'] = pd.to_datetime(df['timestamp'])
df['hour'] = df['timestamp'].dt.hour
df['dayofweek'] = df['timestamp'].dt.dayofweek
df['is_weekend'] = (df['dayofweek'] >= 5).astype(int)

for col in ['transaction_type', 'location', 'device_used', 'payment_channel', 'merchant_category']:
    if col in df.columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))

main_cols = [
    'amount', 'transaction_type', 'hour', 'dayofweek', 'is_weekend',
    'spending_deviation_score', 'velocity_score', 'geo_anomaly_score',
    'location', 'device_used', 'payment_channel', 'merchant_category',
    'time_since_last_transaction',
    'is_fraud'
]
df_clean = df[[c for c in main_cols if c in df.columns]].copy()

df_clean.to_csv("processed_sample.csv", index=False)
joblib.dump(df_clean, OUTPUT)
print("✓ Dane po feature engineering zapisane do processed_sample.csv oraz processed_data.pkl")
