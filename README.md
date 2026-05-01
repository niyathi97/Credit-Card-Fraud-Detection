# 💳 Credit Card Fraud Detection System

## 🚀 Overview

This project is a Machine Learning-based system that detects fraudulent credit card transactions in real-time. It analyzes transaction patterns and predicts whether a transaction is **fraudulent or legitimate**.

This project is designed to be **industry-relevant**, showcasing skills in **data science, machine learning, and deployment with a user interface**.

---

## 🎯 Problem Statement

Credit card fraud is a major issue in digital payments. Fraudulent transactions are rare compared to normal ones, making detection difficult due to **imbalanced datasets**.

---

## 💡 Solution

This system:

* Uses Machine Learning to classify transactions
* Handles imbalanced data using **SMOTE**
* Provides real-time predictions through a **Streamlit UI**
* Displays fraud probability and prediction latency

---

## 🛠️ Tech Stack

* **Python**
* **Pandas, NumPy**
* **Scikit-learn**
* **Imbalanced-learn (SMOTE)**
* **Matplotlib, Seaborn**
* **Streamlit (UI)**
* **Joblib (Model Saving)**

---

## 📂 Project Structure

```
Credit-Card-Fraud-Detection/
│
├── data/               # Dataset (not uploaded due to size)
├── notebooks/          # EDA notebooks
├── src/                # Source code
├── models/             # Saved ML model
├── outputs/            # Results
├── images/             # Graphs & screenshots
├── main.py             # ML pipeline
├── app.py              # Streamlit UI
├── requirements.txt    # Dependencies
└── README.md
```

---

## 📊 Dataset

Dataset used: **Credit Card Fraud Detection Dataset (Kaggle)**
🔗 https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

⚠️ Note: Dataset is not included due to GitHub file size limits.
Place the file in:

```
data/creditcard.csv
```

---

## ⚙️ Installation & Setup

### 1️⃣ Clone the repository

```
git clone https://github.com/YOUR_USERNAME/Credit-Card-Fraud-Detection.git
cd Credit-Card-Fraud-Detection
```

### 2️⃣ Create virtual environment

```
python -m venv venv
venv\Scripts\activate
```

### 3️⃣ Install dependencies

```
pip install -r requirements.txt
```

---

## ▶️ Run the Project

### Run ML pipeline

```
python main.py
```

### Run UI

```
streamlit run app.py
```

---

## 💻 Features

* ✔ Fraud detection using ML
* ✔ Handles imbalanced dataset using SMOTE
* ✔ Real-time prediction
* ✔ User-friendly UI
* ✔ Displays fraud probability
* ✔ Shows prediction latency (ms)

---

## 📈 Results

* Improved fraud detection using balanced dataset
* High recall for fraud class
* Real-time predictions in milliseconds

---

## 🧠 Key Concepts Used

* Imbalanced Data Handling (SMOTE)
* Classification Models (Logistic Regression)
* Confusion Matrix & Evaluation Metrics
* Real-time Prediction Systems

---

## 📸 Screenshots

(Add images from `/images` folder here)

---

## 🎯 Future Improvements

* Deploy as a web application
* Add real-time streaming (Kafka)
* Use advanced models (XGBoost, Deep Learning)
* Add dashboard analytics

---

## 👩‍💻 Author

**Your Name**

---

## ⭐ If you like this project

Give it a ⭐ on GitHub!
