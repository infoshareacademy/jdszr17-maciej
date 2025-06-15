import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

FILE = "financial_fraud_detection_dataset.csv"  # <-- zaktualizowana nazwa pliku

df = pd.read_csv(FILE, nrows=100_000)

print(df.info())
print(df.head())
print(df['is_fraud'].value_counts(normalize=True).rename("proportion"))

# Sprawdzenie kolumny typu fraud
if 'fraud_type' in df.columns:
    print(df['fraud_type'].value_counts(dropna=False))
else:
    print("Brak kolumny 'fraud_type' w pliku!")

# Rozkład kwot
plt.figure(figsize=(8,4))
sns.histplot(df['amount'], bins=100, log_scale=(False, True), kde=True)
plt.title("Rozkład kwoty transakcji")
plt.savefig("eda_amount_hist.png"); plt.close()

# Proporcja fraud/not-fraud
plt.figure(figsize=(4,4))
df['is_fraud'].value_counts().plot.pie(
    autopct='%.2f%%', labels=['Not Fraud', 'Fraud'],
    colors=['#4CAF50','#E53935']
)
plt.title("Udział fraudów")
plt.ylabel('')
plt.savefig("eda_fraud_pie.png"); plt.close()

# Rozkład cech behawioralnych
for col in ['spending_deviation_score', 'velocity_score', 'geo_anomaly_score']:
    if col in df.columns:
        plt.figure(figsize=(7,4))
        sns.histplot(df, x=col, hue='is_fraud', bins=50, kde=True, element='step')
        plt.title(f"Rozkład {col} (fraud vs non-fraud)")
        plt.savefig(f"eda_{col}_hist.png"); plt.close()

print("Wygenerowano wykresy EDA.")
