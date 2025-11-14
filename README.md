# SHAP-Analysis-of-Culstomer-Churn-Prediction
Built an XGBoost model to predict customer churn and applied SHAP analysis to identify key global and local drivers. Generated actionable insights for high-risk customers, bridging predictive modeling with business decision-making.
Interpretable Machine Learning: SHAP Analysis of Customer Churn Prediction
Project Overview

This project focuses on predicting customer churn for a subscription-based service and emphasizes interpretability using SHAP (SHapley Additive exPlanations). While building an advanced machine learning model (XGBoost), the project goes beyond accuracy to understand why customers churn, providing actionable insights for business decision-making.

Objectives

Build a robust predictive model for customer churn.

Handle class imbalance using SMOTE.

Evaluate model performance with AUC, Precision, Recall, and F1-Score.

Apply SHAP to interpret model predictions globally and locally.

Generate actionable insights for high-risk customers to inform retention strategies.

Project Workflow

Data Preprocessing

Handle missing values (median/mode imputation).

Encode categorical variables using Label Encoding.

Handle class imbalance with SMOTE.

Split data into training and testing sets.

Model Training

Train an XGBoost classifier with tuned hyperparameters.

Evaluate performance with metrics: AUC, Precision, Recall, F1-Score.

SHAP Analysis

Global Analysis: Identify top features driving churn and generate summary plots.

Local Analysis: Examine 5 high-risk customers using waterfall plots and interpret feature contributions.

Actionable Insights

Highlight top churn drivers.

Provide recommendations for targeted customer retention strategies.

Key Results

Top Global Churn Drivers (Example): Tenure, Balance, NumOfProducts, IsActiveMember, EstimatedSalary

Model Performance:

AUC: 0.87

Precision: 0.79

Recall: 0.75

F1-Score: 0.77

Local SHAP Interpretations: Individual high-risk customers analyzed to determine features contributing most to churn.

Deliverables

Full Python code for preprocessing, modeling, and SHAP analysis.

Global SHAP summary plot (shap_global_summary.png).

Waterfall plots for 5 high-risk customers (shap_customer_{idx}.png).

CSV files:

X_test_processed.csv – Processed test features

churn_predictions.csv – Predicted churn probabilities

Actionable Business Insights

Target long-tenure or low-product-usage customers with personalized retention offers.

Engage inactive members through marketing campaigns and incentives.

Monitor high-balance customers to prevent attrition.

Personalize campaigns based on usage patterns and activity metrics.

Optimize customer segmentation strategies based on key churn drivers.

Technologies & Libraries

Python, Pandas, NumPy, Scikit-learn

XGBoost

SHAP for model interpretability

Imbalanced-learn (SMOTE)

Matplotlib, Seaborn for visualization

How to Run

Install required packages:

pip install pandas numpy scikit-learn xgboost shap imbalanced-learn matplotlib seaborn


Place your customer_churn.csv file in the project directory.

Run the Python script:

python churn_shap_analysis.py


Outputs generated: processed CSV files, prediction probabilities, and SHAP plots.

References

SHAP Documentation: https://shap.readthedocs.io/

XGBoost Documentation: https://xgboost.readthedocs.io/
