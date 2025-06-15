import streamlit as st
import pandas as pd
import os

st.set_page_config(page_title="Financial Fraud Detection — dashboard", layout="wide")

# ----------- Nagłówek i opis
st.title("💸 Financial Fraud Detection — Data Science Pipeline")
st.markdown("""
Projekt: **Wykrywanie oszustw finansowych na syntetycznym zbiorze 5 milionów transakcji**.

Autor: _Maciej_  
Data: _15-06-2025_
---
""")

# ---------- Sekcja EDA
st.header("1. Eksploracyjna analiza danych (EDA)")

col1, col2, col3 = st.columns(3)
with col1:
    if os.path.exists("eda_amount_hist.png"):
        st.image("eda_amount_hist.png", caption="Rozkład kwot transakcji", use_container_width=True)
with col2:
    if os.path.exists("eda_fraud_pie.png"):
        st.image("eda_fraud_pie.png", caption="Proporcja fraudów w zbiorze", use_container_width=True)
with col3:
    if os.path.exists("eda_velocity_score_hist.png"):
        st.image("eda_velocity_score_hist.png", caption="Rozkład velocity_score (fraud vs non-fraud)", use_container_width=True)

st.markdown("""
- Fraud stanowi tylko ok. **0.24% wszystkich transakcji** (problem bardzo niezbalansowany).
- W próbce najczęstszy typ fraudu to `card_not_present`.
""")

# ---------- Feature Engineering
st.header("2. Feature Engineering")
st.markdown("""
- Wyodrębniono cechy czasowe (godzina, dzień tygodnia, flaga weekendu)
- Zakodowano zmienne kategoryczne: typ transakcji, kategoria, urządzenie, kanał płatności, lokalizacja.
- Wybrano cechy numeryczne i behawioralne do modelowania.
- Przygotowano zbiory `processed_sample.csv` i `processed_data.pkl`.
""")

# ---------- Model i wyniki
st.header("3. Trenowanie modelu i wyniki")
st.markdown("""
- Model: **XGBoostClassifier** z wagą dla klasy fraud.
- ROC AUC: **0.89**  
- Recall (fraud): **0.82**  
- Precision (fraud): **0.04**  
- Accuracy: **0.81**
""")

if os.path.exists("model_feature_importance.png"):
    st.image("model_feature_importance.png", caption="Feature importance — najważniejsze cechy w modelu", use_container_width=True)

# ---------- Confusion matrix
st.subheader("Macierz pomyłek (confusion matrix)")
if os.path.exists("confusion_matrix.png"):
    st.image("confusion_matrix.png", caption="Macierz pomyłek (model XGBoost)", use_container_width=True)
else:
    st.info("Brak pliku confusion_matrix.png — wygeneruj confusion matrix w 3_train.py!")

# ---------- Wnioski
st.header("4. Wnioski i rekomendacje")
st.markdown("""
- Model wykrywa większość fraudów (wysoki recall) — zgodnie z celem projektu.
- **Niski precision** — dużo false positive, typowe przy niezbalansowanych danych.
- **Wysoka ogólna skuteczność (ROC AUC ~0.89)** — model dobrze rozróżnia fraudy od nie-fraudów.
- Możliwości rozwoju:
    - Zaawansowane cechy (rolling window, agregacje po kliencie)
    - Dalsze balansowanie danych (SMOTE, undersampling)
    - Tuning progu klasyfikacji
    - Wyjaśnianie decyzji modelu (SHAP, LIME)
""")

# ---------- Sidebar
st.sidebar.header("Nawigacja")
st.sidebar.markdown("""
- 1. EDA
- 2. Feature Engineering
- 3. Model i wyniki
- 4. Wnioski
""")

# Opcjonalnie: podgląd próbki danych
if os.path.exists("financial_fraud_detection_dataset.csv"):
    if st.sidebar.checkbox("Pokaż próbkę danych"):
        df = pd.read_csv("financial_fraud_detection_dataset.csv", nrows=1000)
        st.dataframe(df.head())
