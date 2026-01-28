# WebSentry – Phishing Detection Chrome Extension 

WebSentry is a machine learning–based Chrome extension developed to detect phishing websites in real time. It analyzes URL-based features and uses supervised machine learning (SVM and XGBoost) through a Flask backend to classify websites as **Safe** or **Phishing/Suspicious**, helping users avoid online scams while browsing.

---

## 📌 Project Background (Why this project exists)
Phishing is one of the most common cyber threats where attackers create fake websites that look real (e.g., banking, e-wallet, email login pages) to steal user credentials. Many users cannot easily recognize phishing websites because the design looks convincing. WebSentry aims to reduce this risk by providing **quick warnings inside the browser** before the user enters sensitive information.

---

## 🎯 Objectives
- Detect phishing websites **in real time** during browsing
- Provide a **browser-based** security solution that is easy to use
- Compare **two supervised ML models** (SVM vs XGBoost) and select the better-performing model
- Improve user awareness by showing a clear classification result in the extension UI

---

## ⭐ Key Features
- **Manual scan:** User clicks “Scan Website” to evaluate the current tab.
- **(Optional) Auto scan:** Automatically scans when the user opens a new website (if enabled in your implementation).
- **Risk classification:** Displays output such as Safe / Phishing / Suspicious based on the model prediction.
- **Real-time backend prediction:** Chrome extension sends the URL to Flask API and receives the result instantly.
- **Lightweight approach:** Uses URL features (fast to compute) instead of heavy page content processing.

---

## 🧠 How WebSentry Works (End-to-End Flow)
1. **User opens a website** in Chrome.
2. WebSentry extension reads the **active tab URL**.
3. The extension sends the URL to the **Flask backend API** (local or hosted).
4. The backend runs **feature extraction** (turns the URL into numerical values).
5. The trained ML model (**SVM or XGBoost**) predicts whether the URL is phishing.
6. The backend returns the prediction to the extension.
7. The extension shows a clear result to the user (e.g., Safe / Phishing).

---

## 🏗 System Architecture
**Frontend (Chrome Extension)**
- Captures active tab URL
- Popup UI for scan results
- Sends request to backend (HTTP)

**Backend (Flask)**
- Receives URL from extension
- Extracts URL features
- Loads trained ML model
- Returns prediction response (JSON)

**Machine Learning**
- Supervised model trained using phishing + legitimate URL datasets
- Two models used for comparison:
  - Support Vector Machine (SVM)
  - XGBoost

---

## 🛠 Technologies Used
- **Python + Flask** (backend API)
- **JavaScript** (Chrome extension logic)
- **HTML/CSS** (extension UI)
- **scikit-learn** (SVM and ML pipeline)
- **XGBoost** (gradient boosting model)
- **joblib** (saving/loading trained models)

---

## 🤖 Machine Learning Models (Why SVM + XGBoost)
### 1) Support Vector Machine (SVM)
SVM works well for classification tasks because it finds a boundary that best separates phishing and legitimate URLs. It is often effective when features are well-designed and the dataset is not extremely noisy.

### 2) XGBoost
XGBoost is a strong ensemble method (gradient boosting). It is known for high performance in classification problems because it learns patterns through multiple decision trees and handles complex feature interactions well.

**Reason for using both:**  
Using multiple models helps compare performance objectively and justify the final choice based on results, not assumptions.

---

## 📊 Dataset
WebSentry was trained and evaluated using **multiple datasets** that contain:
- **Phishing URLs** (public sources such as PhishTank / phishing feeds)
- **Legitimate URLs** (trusted sources and labeled normal URLs)

> Note: Large datasets may not be uploaded to GitHub. A sample dataset can be included for demonstration, while the full dataset can be referenced in the report.

---

## 🧪 Evaluation (How you can report results)
Common metrics used in phishing detection:
- **Accuracy**: overall correctness
- **Precision**: how many predicted phishing are truly phishing
- **Recall**: how many real phishing are detected
- **F1-score**: balance between precision and recall

You can include a small table in the README later, e.g.:
- SVM: Accuracy, Precision, Recall, F1
- XGBoost: Accuracy, Precision, Recall, F1

---

## 📁 Recommended Folder Structure
(Example — adjust to match your repo)



WebSentry/
├── backend/
│ ├── app.py
│ ├── feature_extraction.py
│ ├── model_svm.joblib
│ ├── model_xgb.joblib
│ └── requirements.txt
├── chrome-extension/
│ ├── manifest.json
│ ├── popup.html
│ ├── popup.js
│ ├── background.js
│ └── icons/
├── dataset/
│ └── sample.csv
└── README.md


---

## 🚀 How to Run the Project (Step-by-Step)

### 1) Run the Flask Backend
Open terminal in your backend folder:

```bash
pip install -r requirements.txt
python app.py


By default, Flask may run on something like:

http://127.0.0.1:5000

2) Load the Chrome Extension

Open Chrome → go to chrome://extensions/

Turn ON Developer mode

Click Load unpacked

Select the chrome-extension/ folder

3) Test the Extension

Open any website (safe site)

Click WebSentry icon → click Scan Website

Try a known phishing test URL from your dataset/testing list (for academic testing only)

🔌 Example Backend API (Describe your API)

You can document your endpoint like this (edit to match your actual code):

POST /predict
Request (JSON):

{ "url": "https://example.com/login" }


Response (JSON):

{
  "prediction": "phishing",
  "confidence": 0.92
}

🧪 Testing Summary

WebSentry was tested using:

Safe/legitimate websites to verify low false alarms

Known phishing URLs to verify detection capability

Manual scan functionality via extension popup

Backend connectivity between extension and Flask API

⚠ Limitations

Results depend on dataset quality and how up-to-date phishing samples are

Very new (zero-day) phishing URLs may bypass detection

If the Flask backend is not running or unreachable, scanning will not work

URL-based features may not detect attacks that rely heavily on page content

🔮 Future Improvements

Add deep learning model comparison (optional future work)

Add whitelist/blacklist rules for faster blocking

Deploy Flask backend to cloud (so it works anywhere, not just local)

Improve UI with risk score, explanation, and warning banner

## 👤 Project Information

**Project Title:** WebSentry – Phishing Detection Chrome Extension  

Author:: Musthaq Ahmad Bin Shaik Faizal Hassan  

Supervisor: Noor Hazlina Abdul Mutalib  

✅ Submission Note (For portal: Sharing Repository Link)

If your lecturer asks for the repository link, share:
https://github.com/<your-username>/WebSentry-Phishing-Detection-Chrome-Extension


### What you should delete from your current README
In your current file there are repeated lines like `"# WEBSENTRY-PHISHING-DETECTION-CHROME-EXTENSION"` at the bottom — remove those, and keep **only one clean README**. :contentReference[oaicite:1]{index=1}

If you paste this updated README and tell me your **actual folder names** (backend folder name + extension folder name), I can quickly adjust the structure section so it matches your repo exactly.


