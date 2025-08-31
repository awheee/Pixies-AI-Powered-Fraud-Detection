Pixies: AI-Powered Fraud Detection

Fraudulent financial transactions are a major challenge in today’s digital economy.
This project — Pixies: AI-Powered Fraud Detection — leverages Machine Learning (Random Forest Classifier) to automatically detect suspicious transactions and reduce the risk of fraud.

Our solution was developed as part of the National CyberShield Hackathon 2025, with a focus on simplicity, explainability, and strong performance.

⸻

🏆 Hackathon Details
	•	Hackathon Name: National CyberShield Hackathon 2025
	•	Organizer: Madhya Pradesh Police (as a lead-up to the Cybercrime Investigation and Intelligence Summit – CIIS 2025)
	•	About: The hackathon engages bright young minds to solve emerging cybercrime and digital investigation challenges, in collaboration with law enforcement and federal officials.

Problem Statement Chosen:
AI model for flagging suspicious transactions

⸻

👥 Team Pixies
	•	College: VIT Bhopal University
	•	Year of Study: 3rd Year
	•	Branches (multi-disciplinary): Computer Science Core, AIML, Gaming

Team Members:
	•	Avi Jain (Team Leader) – Reg. No. 23BAI11357
	•	Yash Saxena – Reg. No. 23BCE10699
	•	Naman Singh – Reg. No. 23BAI10024
	•	Shlok Shukla – Reg. No. 23BCG10094
	•	Aarya Kashyap – Reg. No. 23BAI11106

Mentor: Dr. Rizwan ur Rahman

⸻

📂 Project Structure
	•	train.py → Train Random Forest model
	•	evaluate.py → Evaluate model (accuracy, classification report, confusion matrix)
	•	inference.py → Load model & run predictions on new data
	•	requirements.txt → Dependencies
	•	README.md → Documentation
	•	Pixies_fraudDetection.ipynb → Full workflow (Google Colab Notebook)
	•	images/ → Graphs & visualizations

⸻

📊 Dataset & Model Files

Due to large file sizes, the datasets and trained model are stored externally.
You can download them from Google Drive:

📂 Dataset & Model: Google Drive Link
	•	fraudTrain.csv – Training dataset
	•	fraudTest.csv – Testing dataset
	•	fraud_detection_model.pkl – Trained Random Forest model

⸻

🚀 Usage
	•	Run train.py → trains the Random Forest model
	•	Run evaluate.py → evaluates accuracy, precision, recall, and generates graphs
	•	Run inference.py → uses the saved model to predict fraud on new data

⸻

✅ Results

Our Random Forest model achieved high overall accuracy but with some trade-offs between fraud and non-fraud detection:
	•	Accuracy: 0.9973
	•	Precision (Fraud): 0.78
	•	Recall (Fraud): 0.45
	•	F1-Score (Fraud): 0.57

Classification Report:

               precision    recall  f1-score   support

           0       1.00      1.00      1.00    553574
           1       0.78      0.45      0.57      2145

    accuracy                           1.00    555719
   macro avg       0.89      0.72      0.78    555719
weighted avg       1.00      1.00      1.00    555719

⚠️ Note: The model performs extremely well for non-fraud transactions (majority class) but detecting fraud cases (minority class) remains more challenging.

⸻

📈 Visualizations
	•	Fraud vs Non-Fraud Distribution (images/fraud_vs_nonfraud.png)
	•	Transaction Amount Distribution (images/amount_distribution.png)
	•	Fraud Cases Over Time (images/fraud_over_time.png)
	•	Top 15 Feature Importances (images/feature_importance.png)
	•	Confusion Matrix (images/confusion_matrix.png)

⸻

🔮 Future Improvements
	•	Apply SMOTE / class balancing techniques to better handle fraud class imbalance.
	•	Try advanced models like XGBoost or Neural Networks.
	•	Deploy the model as a real-time fraud detection API.
	•	Incorporate explainability (SHAP values, LIME) for better trust in decisions.

⸻

✨ Team Pixies – Building for a Safer Digital Future

Built with 💡 and ☕ during the National CyberShield Hackathon 2025.

⸻
