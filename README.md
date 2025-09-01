# Pixies: AI-Powered Fraud Detection  

Fraudulent financial transactions are a major challenge in today’s digital economy.  
This project — **Pixies: AI-Powered Fraud Detection** — leverages Machine Learning models (**Logistic Regression, Decision Tree, Random Forest, XGBoost**) to automatically detect suspicious transactions and reduce the risk of fraud.  

Our solution was developed as part of the **National CyberShield Hackathon 2025**, with a focus on simplicity, explainability, and strong performance.  

---

## 🏆 Hackathon Details  

- **Hackathon Name:** National CyberShield Hackathon 2025  
- **Organizer:** Madhya Pradesh Police (as a lead-up to the Cybercrime Investigation and Intelligence Summit – CIIS 2025)  
- **About:** The hackathon engages bright young minds to solve emerging cybercrime and digital investigation challenges, in collaboration with law enforcement and federal officials.  

**Problem Statement Chosen:**  
*AI model for flagging suspicious transactions*  

---

## 👥 Team Pixies  

- **College:** VIT Bhopal University  
- **Year of Study:** 3rd Year  
- **Branches (multi-disciplinary):** Computer Science Core, AIML, Gaming  

**Team Members:**  
- **Avi Jain (Team Leader)** – Reg. No. 23BAI11357  
- Yash Saxena – Reg. No. 23BCE10699  
- Naman Singh – Reg. No. 23BAI10024  
- Shlok Shukla – Reg. No. 23BCG10094  
- Aarya Kashyap – Reg. No. 23BAI11106  

**Mentor:** Dr. Rizwan ur Rahman  

---

## 📂 Project Structure  

- **train_logistic_regression.py** → Train Logistic Regression model  
- **train_decision_tree.py** → Train Decision Tree model  
- **train_random_forest.py** → Train Random Forest model  
- **train_xgboost.py** → Train XGBoost model  
- **evaluate_models.py** → Compare all models (accuracy, precision, recall, F1)  
- **inference.py** → Load trained model & run predictions on new data  
- **requirements.txt** → Dependencies  
- **README.md** → Documentation  
- **Pixies_fraudDetection.ipynb** → Full workflow (Google Colab Notebook)  
- **images/** → Graphs & visualizations  

---

## 📊 Dataset & Model Files  

Due to large file sizes, the datasets and trained models are stored externally.  
You can download them from Google Drive:  

📂 Dataset & Models: [Google Drive Link](https://drive.google.com/drive/folders/1J1uMKDamSZ5UDbXQbo-2OfqlYo0qd_cB?usp=sharing)  

- **fraudTrain.csv** – Training dataset  
- **fraudTest.csv** – Testing dataset  
- **logistic_regression_model.pkl** – Trained Logistic Regression model  
- **decision_tree_model.pkl** – Trained Decision Tree model  
- **random_forest_model.pkl** – Trained Random Forest model  
- **xgboost_model.pkl** – Trained XGBoost model  

---

## 🚀 Usage  

1. **Train Models:**  
   ```bash
   python train_logistic_regression.py
   python train_decision_tree.py
   python train_random_forest.py
   python train_xgboost.py
   ```

2. **Evaluate All Models Together:**  
   ```bash
   python evaluate_models.py
   ```

3. **Run Inference on New Data:**  
   ```bash
   python inference.py
   ```

---

## 📊 Results  

### Individual Model Performance  

- **Logistic Regression:** Accuracy **95.6%**, but very weak fraud detection (fraud recall = 0).  
- **Decision Tree:** Accuracy **99.66%**, weak recall for fraud.  
- **Random Forest:** Accuracy **99.66%**, strong overall, but still misses frauds.  
- **XGBoost:** Accuracy **99.54%**, balanced macro recall compared to others.  

### Model Comparison Table  

| Model                | Accuracy | Fraud Precision | Fraud Recall | Fraud F1 | Macro Precision | Macro Recall | Macro F1 |
|----------------------|----------|-----------------|--------------|----------|-----------------|--------------|----------|
| Logistic Regression  | 0.9561   | 0.00            | 0.00         | 0.00     | 0.53            | 0.85         | 0.54     |
| Decision Tree        | 0.9966   | 0.00            | 0.00         | 0.00     | 0.78            | 0.77         | 0.77     |
| Random Forest        | 0.9966   | 0.00            | 0.00         | 0.00     | 0.99            | 0.56         | 0.61     |
| XGBoost              | 0.9954   | 0.00            | 0.00         | 0.00     | 0.71            | 0.81         | 0.75     |

⚠️ **Observation:**  
- All models achieve very high overall accuracy because the dataset is highly imbalanced (fraud cases are very rare).  
- Fraud detection (class 1) remains the biggest challenge: all models fail to recall fraud cases effectively.  
- Macro scores reveal that **XGBoost and Decision Tree handle class imbalance slightly better**.  

---

## 📈 Visualizations  

- Fraud vs Non-Fraud Distribution (`images/fraud_vs_nonfraud.png`)
<img width="567" height="455" alt="fraud_vs_nonfraud" src="https://github.com/user-attachments/assets/9ce93436-6636-43e8-8830-be64797f3034" />


- Transaction Amount Distribution (`images/amount_distribution.png`)  
<img width="582" height="455" alt="amount_distribution" src="https://github.com/user-attachments/assets/6cff6fb6-2a65-4f28-91c7-004ecbda4533" />


- Fraud Cases Over Time (`images/fraud_over_time.png`)  
<img width="576" height="455" alt="fraud_over_time" src="https://github.com/user-attachments/assets/840a65db-3e29-4504-9ecf-84c9dfd94d58" />


- Top 15 Feature Importances (`images/feature_importance.png`)
<img width="738" height="435" alt="feature_importance" src="https://github.com/user-attachments/assets/43a761c8-3063-4d3d-b7ae-133b5b1b3bbf" />


- Confusion Matrix (`images/confusion_matrix.png`)  
<img width="565" height="455" alt="confusion_matrix" src="https://github.com/user-attachments/assets/7e17e677-ddc1-4d1d-9b4e-64073c7ce536" />

- Model comparision
<img width="846" height="637" alt="image" src="https://github.com/user-attachments/assets/7b049a7f-eb17-438b-8807-30d5f9676def" />

---

## 🔮 Future Improvements  

- Apply **SMOTE / class balancing** techniques to improve fraud detection.  
- Explore **Neural Networks and Ensemble Methods**.  
- Deploy the model as a **real-time fraud detection API**.  
- Add **explainability (SHAP, LIME)** for better trust in predictions.  

---

## ✨ Team Pixies – Building for a Safer Digital Future  

Built with 💡 and ☕ during the **National CyberShield Hackathon 2025**.  
