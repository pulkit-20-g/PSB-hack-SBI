# PSB-hack-SBI
# 🧠 Defaulter Prediction System - SBI Hackathon Project

This project is a comprehensive machine learning pipeline built to **predict loan defaulters** using customer banking, credit bureau, and behavioral data. Developed as part of a **State Bank of India Hackathon**, the system uses a **meta-modeling ensemble** of neural networks and LightGBM to achieve high recall while balancing precision.



---

## 🚀 Features

- 🔄 Extensive preprocessing: null handling, categorical transformation, flag conversion, and outlier clipping
- 🧮 Neural network with class imbalance handling (AUC + recall monitored)
- 🌲 LightGBM model for fast structured learning
- 🧠 Meta-model using Logistic Regression to combine predictions
- 📈 Stratified k-fold cross-validation (5-fold)
- 🎯 Precision-Recall tradeoff with threshold tuning to hit target recall

---

## 🛠 Tech Stack

- **Python 3**, **Pandas**, **NumPy**, **Scikit-learn**
- **TensorFlow / Keras** (Neural Net)
- **LightGBM**
- **Matplotlib**, **Seaborn** (EDA, plotting)
- **joblib** (model saving)

---

## 🧩 Model Pipeline

1. **Data Cleaning**
   - Date string conversions, encoding income bands
   - One-hot encoding for product/group types
   - Imputation: 0 for transaction fields, median for bureau/numeric
   - Drop high-null fields and sparse categories

2. **Modeling**
   - Neural Network (3 hidden layers, dropout, class weights)
   - LightGBM with balanced class weights
   - Meta-model: Logistic Regression trained on out-of-fold predictions
   - Threshold tuning based on Precision-Recall curve

3. **Evaluation**
   - Confusion matrix + classification report
   - ROC-AUC, accuracy, precision, recall per fold

---

## 📦 Outputs

- `defaulter_model.h5` – Trained neural network
- `defaulter_blend.pkl` – Full stacked ensemble with scaler and optimal threshold
- `confusion_matrix.png` – Final evaluation heatmap

---

## 📂 Folder Structure

