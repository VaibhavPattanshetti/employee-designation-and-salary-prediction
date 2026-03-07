# 📊 Employee Designation & Salary Prediction

A **machine learning-based web application** to predict **employee salary** and **job designation** based on personal, professional, and performance-related factors.
This project allows HR professionals, students, and enthusiasts to **estimate potential salary and designation** and make **data-driven decisions**.

---

## 🔗 Live Demo

Try the live app here: 👉 [Employee Designation & Salary Prediction App](https://employee-designation-and-salary-prediction-4dxracux9jersbac2xl.streamlit.app/)

---

## 📖 Project Description

Efficient employee productivity analysis helps in:

- **Predicting salary** based on experience, skills, education, and performance.
- **Predicting job designation** based on salary, experience, and productivity score.
- **Supporting HR decisions** with ML-powered insights.

This web app takes in employee details like age, gender, experience, education, skills, department, and performance metrics, then predicts the **expected salary** and **job designation**.

---

## 📊 Key Results

| Task | Model | Performance on test data |
|----|----|----|
| Salary Prediction | Lasso | R² ≈ 0.90 |
| Role Prediction | SVM | ~71% Accuracy |

---

## 🗃️ Dataset

The model was trained on a custom **Employee Dataset** (`employee_dataset.csv`) included in this repository.

**Dataset Features:**

- Age
- Gender (Male/Female)
- Designation (Junior, Executive, Lead, Senior, Manager)
- Experience (Years)
- Productivity Score (1–100)
- Education Level (High School, Bachelor, Master, PhD)
- Performance Rating (1–5)
- Last Promotion Year
- Work Location (Bangalore, Chennai, Delhi, Hyderabad, Kolkata, Mumbai, Noida, Pune)
- Department (Finance, HR, IT, Marketing, Operations, Sales)
- Skillset (20 skills including Python, SQL, Data Analysis, etc.)

**Target Variables:**
- Salary (Regression)
- Designation (Classification)

---

## 🤖 Machine Learning Models

### 💰 Salary Prediction
- **Model Used:** Lasso Regression
- **Why Lasso?**
  - Handles feature selection automatically
  - Reduces overfitting via regularization
  - Works well with high-dimensional data (40 features)

### 🏷️ Designation Prediction
- **Model Used:** SVM (RBF Kernel)
- **Why SVM?**
  - Effective for multi-class classification
  - Works well on small to medium datasets
  - RBF kernel handles non-linear boundaries

**Steps followed:**
1. Preprocessed the dataset (handled categorical and numerical features)
2. Applied Ordinal Encoding, One-Hot Encoding, and MultiLabelBinarizer
3. Split into train/test sets
4. Trained Lasso Regression and SVM models
5. Saved the models as pickle (`.pkl`) files for deployment

**Additional Info:**
- Multiple ML models were trained and evaluated; Lasso Regression and SVM (RBF Kernel) were selected as the best performing models for salary and designation prediction respectively.
- This repository includes the training notebook: `Notebook.ipynb`
- Trained models (`lasso_model.pkl`, `svm_model.pkl`) are included for direct use
- App falls back to rule-based estimation if model files are unavailable

---

## 🛠️ Tech Stack

- **Language:** Python
- **Frontend:** Streamlit
- **ML Libraries:** Scikit-learn, NumPy, Pandas
- **Model Serialization:** Joblib

---

## 🚀 How to Run Locally

1. **Clone the repository**
```bash
git clone https://github.com/VaibhavPattanshetti/employee-designation-and-salary-prediction.git
cd employee-designation-and-salary-prediction
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Run the app**
```bash
streamlit run app.py
```
---

## 🐳 Docker

1. **Pull image from Docker Hub**
```bash
docker pull justvaibhav/employee-predictor:latest
```

2. **runs/starts the downloaded image as a container**
```bash
docker run -p 8501:8501 justvaibhav/employee-predictor:latest
```

Then open http://localhost:8501 in your browser.

**Or Build Locally**
```bash
docker build -t employee-predictor .
docker run -p 8501:8501 employee-predictor
```

Docker Hub image here: 👉 [justvaibhav/employee-predictor](https://hub.docker.com/r/justvaibhav/employee-predictor)

---

## 📁 Project Structure

```
employee-designation-and-salary-prediction/
├── app.py                  # Main Streamlit application
├── lasso_model.pkl         # Trained Lasso Regression model
├── svm_model.pkl           # Trained SVM model
├── employee_dataset.csv    # Dataset used for training
├── Notebook.ipynb          # Training notebook
├── requirements.txt        # Python dependencies
├── .gitignore              # Git ignore file
├── LICENSE                 # MIT License
└── README.md               # Project documentation
```

---
