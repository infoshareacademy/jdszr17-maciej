import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, ConfusionMatrixDisplay
from xgboost import XGBClassifier
import matplotlib.pyplot as plt

df = joblib.load("processed_data.pkl")

X = df.drop(['is_fraud'], axis=1)
y = df['is_fraud']

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)

scale_pos_weight = (len(y_train) - sum(y_train)) / sum(y_train)
model = XGBClassifier(
    n_estimators=200, max_depth=6, scale_pos_weight=scale_pos_weight,
    learning_rate=0.1, n_jobs=-1, use_label_encoder=False, eval_metric='logloss'
)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:,1]

print(classification_report(y_test, y_pred))
print("ROC AUC:", roc_auc_score(y_test, y_proba))
print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))

# Feature importance
importances = model.feature_importances_
feat_names = X.columns
plt.figure(figsize=(8,5))
plt.barh(feat_names, importances)
plt.title("Feature importance (XGBoost)")
plt.tight_layout()
plt.savefig("model_feature_importance.png"); plt.close()
print("✓ Feature importance zapisane do model_feature_importance.png")

# --- ZAPISZ CONFUSION MATRIX JAKO PNG ---
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
fig, ax = plt.subplots(figsize=(5, 5))
disp.plot(cmap=plt.cm.Blues, ax=ax, colorbar=False)
plt.title("Macierz pomyłek (confusion matrix)")
plt.tight_layout()
plt.savefig("confusion_matrix.png")
plt.close()
print("✓ Confusion matrix zapisana do confusion_matrix.png")
