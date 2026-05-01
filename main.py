import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE

# -----------------------------
# LOAD DATA
# -----------------------------
df = pd.read_csv("data/creditcard.csv")

# -----------------------------
# FEATURES & TARGET
# -----------------------------
X = df.drop("Class", axis=1)
y = df["Class"]

# -----------------------------
# TRAIN-TEST SPLIT
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -----------------------------
# APPLY SMOTE
# -----------------------------
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

print("After SMOTE:")
print(y_train_res.value_counts())

# -----------------------------
# TRAIN MODEL
# -----------------------------
model = LogisticRegression(max_iter=1000)
model.fit(X_train_res, y_train_res)

# -----------------------------
# PREDICTIONS
# -----------------------------
y_pred = model.predict(X_test)

# -----------------------------
# EVALUATION
# -----------------------------
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:\n")
print(confusion_matrix(y_test, y_pred))

import matplotlib.pyplot as plt
import seaborn as sns

# -----------------------------
# CONFUSION MATRIX VISUALIZATION
# -----------------------------
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')

plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.savefig("images/confusion_matrix.png")
plt.show()

import joblib

# -----------------------------
# SAVE MODEL
# -----------------------------
joblib.dump(model, "models/fraud_model.pkl")
print("\n✅ Model saved successfully!")

# -----------------------------
# LOAD MODEL
# -----------------------------
loaded_model = joblib.load("models/fraud_model.pkl")

# -----------------------------
# SAMPLE PREDICTION
# -----------------------------
sample = X_test.iloc[0:1]

prediction = loaded_model.predict(sample)
print("\nSample Prediction (0=Normal, 1=Fraud):", prediction)