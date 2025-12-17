# Customer Churn Prediction using Machine Learning

This project focuses on predicting customer churn for a subscription-based telecom company.  
By identifying customers at risk of leaving, businesses can take proactive retention measures.

---

## Problem Statement

Customer churn is a major challenge for subscription-based businesses, as losing customers directly impacts revenue.  
The goal of this project is to build a **machine learning model** that predicts whether a customer is likely to churn based on demographic information, service usage, and billing details.  

By predicting churn early, companies can implement targeted strategies to retain high-risk customers.

---

## Dataset Description

The dataset used is the **Telco Customer Churn Dataset**, which includes:

- **Demographic features**: gender, senior citizen status, partner, dependents  
- **Service features**: phone service, multiple lines, internet service, online security, online backup, device protection, tech support, streaming TV/movies  
- **Contract and billing features**: contract type, paperless billing, payment method, tenure, monthly charges, total charges  
- **Target variable**: `Churn` (Yes/No)

The dataset contains **7043 customer records** and **20+ features**.

---

## Model Pipeline

The machine learning pipeline includes the following steps:

1. **Data Cleaning**
   - Convert `TotalCharges` to numeric  
   - Remove missing or inconsistent entries  
   - Drop unnecessary columns like `customerID`

2. **Feature Engineering**
   - Identify **numerical** and **categorical** columns
   - Encode categorical features using **OneHotEncoder**
   - Scale numerical features using **StandardScaler**

3. **Pipeline & Model**
   - Preprocessing combined into a **`ColumnTransformer`**
   - Logistic Regression classifier wrapped inside a **Pipeline**
   - Handle class imbalance with `class_weight='balanced'`
   - Train-test split used to validate performance

---

## Evaluation Metrics

The model was evaluated on a test set using:

- **Accuracy**: 0.79  
- **ROC-AUC score**: 0.83  
- **Precision, Recall, F1-score**:  
