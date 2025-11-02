# 🏦 Home Credit Default Risk - Machine Learning Pipeline

## 📘 Project Overview

This project implements an **end-to-end machine learning pipeline** for predicting default probability in the **Home Credit Default Risk** dataset.  
It follows a modular, step-by-step structure — from data exploration to model deployment — allowing easy learning and extension.

The goal is to build **robust, interpretable models** (LightGBM and Logistic Regression) that integrate multiple relational tables while maintaining best practices for reproducibility and explainability.

---

## 🧩 Pipeline Structure

### **1️⃣ Data Loading**

- Reads all relevant CSVs from the Home Credit dataset
- Uses memory-efficient dtypes to optimize loading speed

### **2️⃣ Exploratory Data Analysis (EDA)**

- Summarizes dataset size, missing values, target distribution
- Identifies skewed and correlated features
- Visualizes key numeric distributions for interpretability

### **3️⃣ Data Cleaning**

- Handles missing values (`NaN`, outliers, and anomalies)
- Converts categorical columns to consistent types
- Caps extreme outliers for numeric stability

### **4️⃣ Feature Engineering & Aggregation**

- Aggregates relational tables by `SK_ID_CURR` (main key)
- Computes statistical summaries (mean, sum, count, max, etc.)
- Joins results into the main application dataset
- Extracts domain features such as credit utilization ratios and payment patterns

### **5️⃣ Encoding & Splitting**

- Encodes categorical variables with appropriate encoding strategies
- Splits training and test sets
- Prepares folds for cross-validation (Stratified KFold by target)

### **6️⃣ Modeling (LightGBM & Logistic Regression)**

- Implements **OOF (Out-Of-Fold)** training for both models
- Logs CV scores and feature importances
- Stores trained models and metrics for reproducibility

### **7️⃣ Model Interpretation**

- Uses **SHAP** to explain top contributing features for LightGBM
- Generates coefficient analysis for Logistic Regression
- Creates comprehensive visualization plots

### **8️⃣ Model Saving**

- Saves model weights (`.pkl`) and feature lists
- Stores OOF predictions and test predictions for both models

### **9️⃣ Results & Analysis**

- Generates performance metrics and visualizations
- Creates feature importance rankings for both models
- Produces comprehensive model evaluation reports

---

## 📁 Project Structure

```
HOME_CREDIT_PROJECT/
│
├── models/
│   ├── features_list.pkl
│   ├── final_model_v1.pkl
│   ├── oof_preds_lgb.pkl
│   ├── oof_preds_lr.pkl
│   ├── test_preds_lgb.pkl
│   └── test_preds_lr.pkl
│
├── output/
│   ├── plots/
│   │   ├── target_distribution.png
│   │   ├── missing_values_top30.png
│   │   ├── numeric_distributions.png
│   │   ├── lgbm_top40_feature_importance.png
│   │   ├── lgbm_score_distribution_by_class.png
│   │   ├── lr_top30_coef_magnitude.png
│   │   ├── roc_curve_oof.png
│   │   ├── pr_curve_oof.png
│   │   ├── shap_summary_top30.png
│   │   └── shap_dependence_EXT_SOURCE_3.png
│   │
│   └── result/
│       ├── feature_importance_lgb.csv
│       └── feature_importance_lr.csv
│
├── main.ipynb
└── README.md
```

---

## 📈 Key Outputs

- **Model Performance**: Comparative analysis between LightGBM and Logistic Regression
- **Top Features**:
  - LightGBM: External sources, credit history, payment behavior
  - Logistic Regression: Income-related features, loan characteristics
- **Model Interpretation**: SHAP analysis and coefficient magnitude plots
- **Data Quality**: Missing value analysis and target distribution

---

## 🧾 License

This project is for educational and research purposes.
You are free to adapt or extend it for Kaggle competition or production prototyping.

---

## 👨‍💻 Author

**Nafis**  
Machine Learning & Data Science Enthusiast
