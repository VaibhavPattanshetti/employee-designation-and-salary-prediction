# 📊 Suitable Package & Role Prediction using Machine Learning

A **machine learning–based employee analytics system** that predicts **salary packages and suitable job roles/designations** based on employee productivity, experience, skills, and performance metrics.

This project demonstrates an **end-to-end ML pipeline**, including **data preprocessing, EDA, feature engineering, multiple regression and classification models, evaluation, and comparison**.

---

## 🚀 Project Highlights
- 📈 **Salary (Package) Prediction** using regression models
- 🧑‍💼 **Role / Designation Prediction** using classification models
- 🧹 Real-world **data cleaning & preprocessing**
- 📊 Detailed **EDA and feature analysis**
- 🔍 Model comparison using industry-standard metrics
- 🎯 Focus on **generalization and model interpretability**

---

## 📌 Project Overview

Organizations often struggle to:
- Estimate **fair salary packages**
- Assign **suitable roles/designations**
- Evaluate employees objectively based on multiple factors

This project uses **machine learning techniques** to analyze employee data and:
- Predict **salary packages**
- Recommend **appropriate job roles**
- Assist **HR decision-making** using data-driven insights

---

## 🗂 Dataset

**Type:** Structured tabular employee dataset  

### Key Features:
- Age
- Gender
- Department
- Designation
- Experience (Years)
- Skillset
- Productivity Score
- Performance Rating
- Education Level
- Work Location
- Last Promotion Year
- Salary (Target variable for regression)

### Dataset Characteristics:
- Real-world noisy data
- Missing values
- Categorical + numerical features
- Presence of outliers

---

## 🛠 Tech Stack

| Category | Tools |
|--------|------|
| Language | Python |
| Data Analysis | NumPy, Pandas |
| Visualization | Matplotlib, Seaborn |
| Machine Learning | Scikit-learn |
| Feature Engineering | Encoding, Scaling |
| Platform | Jupyter Notebook / Google Colab |

---

## 🔄 Project Workflow

Data Collection
↓
Data Cleaning & Preprocessing
↓
Exploratory Data Analysis (EDA)
↓
Feature Engineering
↓
Model Training
↓
Model Evaluation & Comparison
↓
Prediction on New Data

---


---

## 🧹 Data Preprocessing

- Dropped non-informative columns (IDs, names, dates)
- Handled missing values:
  - Numerical → Mean imputation
  - Categorical → Mode imputation
- Removed duplicate records
- Outlier analysis using **IQR method**
- Converted categorical variables using:
  - Label Encoding
  - One-Hot Encoding
  - Ordinal Encoding
- Skillset processing using **MultiLabelBinarizer**

---

## 📊 Exploratory Data Analysis (EDA)

- Distribution analysis of numerical features
- Boxplots for outlier detection
- Correlation heatmap for feature relationships
- Categorical feature distribution using bar & pie charts
- Salary distribution and skewness analysis

---

## 📈 Machine Learning Models

### 🔹 Regression Models (Salary Prediction)
- Multiple Linear Regression
- Polynomial Regression
- Ridge Regression
- Lasso Regression
- ElasticNet Regression

### 🔹 Classification Models (Role Prediction)
- Decision Tree Classifier
- Hyperparameter tuning using GridSearchCV

---

## 🧮 Evaluation Metrics

### Regression Metrics:
- **R² Score**
- Mean Absolute Error (MAE)
- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)

### Classification Metrics:
- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix

---

## 🏆 Model Performance Summary

### 🔹 Regression Results
- Best models achieved **~90% R² score**
- Consistent training and testing performance
- Low MAE and RMSE → good generalization

### 🔹 Classification Results
- Initial Decision Tree showed overfitting
- Improved performance after **hyperparameter tuning**
- Balanced evaluation using **macro & weighted averages**

---

## 🔍 Key Insights

- Experience and productivity strongly influence salary
- Certain skills significantly impact compensation
- Overfitting can occur without proper tuning
- Regularization improves model stability
- Feature selection improves classification accuracy

---

## 🔮 Future Improvements
- Add ensemble models (Random Forest, XGBoost)
- Deploy as a web application for HR teams
- Add explainability using SHAP / feature importance
- Extend dataset with real organizational data

---

## ⚠️ Disclaimer
This project is intended for **educational and analytical purposes only** and should not be used as a standalone HR decision system without expert validation.

---

## 👤 Author
**Vaibhav Pattanashhetti**  
B.Tech – Computer Engineering  
Interested in **Machine Learning, Data Analytics, and Applied AI**

---

⭐ If you find this project useful, consider starring the repository!

## 🔄 Project Workflow

