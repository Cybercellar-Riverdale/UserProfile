# **UserProfile for Spam Email Detection**

## **Overview**
This project is designed to analyze email content, assess risk scores, and detect spam or phishing attempts. It integrates **user profiling**, **risk assessment**, **machine learning**, and **LLM-based attachment analysis** to enhance email security.

---

## **Project Phases**

### **Phase 1: User Profiling**
- **Goal:** Analyze user profiles based on login data, reports, and risk assessments.
- **Technologies:** `pandas`, `numpy`, `json`, `matplotlib`, `sklearn`
- **Key Functions:**
  - `get_latest_login(email)`: Fetches the most recent login data.
  - `get_latest_report(email)`: Retrieves the latest report details.
  - `generate_profile(email)`: Aggregates login, report, and risk data into a user profile.
- **Data Sources:** JSON (`login_merged.json`, `reports_response_merge.json`, `risk_summary.json`) and CSV (`enhanced_risk_assessment_results.csv`).

---

### **Phase 2: Risk Score Analysis**
- **Goal:** Assign risk scores to emails based on grammar, sentiment, keywords, and punctuation.
- **Technologies:** `pandas`, `re`, `VADER Sentiment Analysis`, `TextBlob`, `language_tool_python`
- **Risk Calculation:**
  - Warning words (30 points)
  - Exclamation marks (20 points)
  - Suspicious phrases (50 points)
- **Risk Categories:**
  - **Low (0-30):** Safe
  - **Medium (31-60):** Possibly suspicious
  - **High (61-100):** Requires urgent attention
- **Output:** Risk scores stored in `enhanced_risk_assessment_results.csv` and summary in `risk_summary.json`.

---

### **Phase 3: ML Model Integration for Risk Prediction**
- **Goal:** Train an ML model to predict email risk scores.
- **Technologies:** `pandas`, `numpy`, `scikit-learn`, `XGBoost`, `joblib`
- **Workflow:**
  - Preprocess data (fill missing values, encode labels, scale features).
  - Train an `XGBoost` classifier on email risk assessment data.
  - Use `Stratified K-Fold` cross-validation for evaluation.
  - Store trained models using `joblib`.
- **Features Used:** `warning_word_score`, `exclamation_score`, `spam_word_score`, etc.

---

### **Phase 4: Attachment Analysis Using LLM**
- **Goal:** Detect spam/phishing threats in email attachments.
- **Technologies:** `DistilBERT`, `pandas`, `numpy`, `torch`, `PyPDF2`, `pytesseract`, `docx`
- **Workflow:**
  - Extract text from `txt`, `docx`, `pdf`, `csv`, `xlsx`, and image files.
  - Preprocess text (lowercase conversion, special character removal).
  - Use **DistilBERT** for AI-based classification.
  - Combine **ML predictions** with **keyword-based filtering** for improved spam detection.
- **Why Hybrid Approach?**
  - **AI Model Prediction:** Context-aware detection of spam.
  - **Keyword-Based Filtering:** Flags common spam words.
  - **Combined Approach:** Reduces false positives and improves accuracy.

---

## **Installation & Usage**
### **Prerequisites**
- Python 3.x
- Required libraries:  
  ```sh
  pip install pandas numpy scikit-learn xgboost vaderSentiment language-tool-python transformers torch pytesseract PyPDF2 python-docx fastapi==0.110.0 pydantic==2.6.3 uvicorn==0.29.0 joblib==1.3.2 numpy==1.26.2 scikit-learn==1.3.2 xgboost==2.0.3 jsonschema==4.19.2

  
