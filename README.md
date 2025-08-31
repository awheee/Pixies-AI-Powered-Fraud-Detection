# Pixies: AI-Powered Fraud Detection  

Fraudulent financial transactions are a major challenge in today’s digital economy.  
This project — **Pixies: AI-Powered Fraud Detection** — leverages Machine Learning (Random Forest Classifier) to automatically detect suspicious transactions and reduce the risk of fraud.  

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

- **train.py** → Train Random Forest model  

- **evaluate.py** → Evaluate model (accuracy, classification report, confusion matrix)  

- **inference.py** → Load model & run predictions on new data  

- **requirements.txt** → Dependencies  

- **README.md** → Documentation  

- **Pixies_fraudDetection.ipynb** → Full workflow (Google Colab Notebook)  

- **images/** → Graphs & visualizations  


---

## 📊 Dataset & Model Files  

Due to large file sizes, the datasets and trained model are stored externally.  
You can download them from Google Drive:  

📂 Dataset & Model: [Google Drive Link](https://drive.google.com/drive/folders/1J1uMKDamSZ5UDbXQbo-2OfqlYo0qd_cB?usp=sharing)  


- **fraudTrain.csv** – Training dataset  

- **fraudTest.csv** – Testing dataset  

- **fraud_detection_model.pkl** – Trained Random Forest model  


---

## 🚀 Usage  

- Run **train.py** → trains the Random Forest model  

- Run **evaluate.py** → evaluates accuracy, precision, recall, and generates graphs  

- Run **inference.py** → uses the saved model to predict fraud on new data  


---

## 📊 Results

- Model Accuracy: **99.7%**  
- The model performs extremely well in detecting non-fraud transactions.  
- For fraud cases, it achieved **78% precision** and **45% recall**, meaning it can identify many fraudulent activities but still misses some cases.  
- Overall, the Random Forest model shows strong performance with room for improvement in recall for fraud detection.

⚠️ Note: The model performs extremely well for non-fraud transactions (majority class) but detecting fraud cases (minority class) remains more challenging.  


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


---

## 🔮 Future Improvements  

- Apply SMOTE / class balancing techniques to better handle fraud class imbalance.  

- Try advanced models like XGBoost or Neural Networks.  

- Deploy the model as a real-time fraud detection API.  

- Incorporate explainability (SHAP values, LIME) for better trust in decisions.  


---

## ✨ Team Pixies – Building for a Safer Digital Future  

Built with 💡 and ☕ during the **National CyberShield Hackathon 2025**.  
