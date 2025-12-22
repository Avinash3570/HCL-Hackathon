# HCL-Hackathon

## HCL Tech Hackathon

**Git Repo** - https://github.com/Avinash3570/HCL-Hackathon.git

**Dataset** - https://www.kaggle.com/datasets/sahilislam007/online-retail-customer-churn-prediction-dataset

**streamlit deployed app** - https://churn-prediction-rwpcgtetaohbm8cjdtazli.streamlit.app/ 

**app-github link** - https://github.com/swakshpatwari05/Churn-Prediction
## Procedure:


**Work Flow Design**

```
Raw Customer Data (CSV)
        ↓
Data Validation
        ↓
Preprocessing & Encoding
        ↓
Train-Test Split
        ↓
Model Training
   ├─ Logistic Regression
   ├─ Random Forest
   └─ XGBoost
        ↓
Model Evaluation
        ↓
Model Comparison
        ↓
Final Model Selection
        ↓
Prediction / Deployment
```

## Data Cleaning & Preprocessing Workflow

**Input:** Raw Customer Data (CSV)

1. **Data Ingestion**
   - Load dataset
   - Validate schema and column consistency
   - Identify target variable Churn

2. **Identifier Handling**
   - Detect unique identifiers (CustomerID)
   - Exclude identifiers from modeling features
   - Retain only for traceability

3. **Data Type Standardization**
   - Convert numerical attributes to numeric types
   - Convert Last_Purchase_Date to datetime
   - Validate Churn as binary (0/1)

4. **Missing Value Treatment**
   - Detect missing values per column
   - Numerical features → median imputation
   - Categorical features → mode / "Unknown"

5. **Data Quality Validation**
   - Detect invalid values (age, income, purchase metrics)
   - Handle outliers (cap / remove)
   - Ensure logical value ranges

6. **Categorical Encoding**
   - Binary categories → label encoding
   - Multi-class categories → one-hot encoding
   - Generate numerical feature matrix

7. **Feature Scaling**
   - Select numerical features
   - Apply standardization (StandardScaler)
   - Preserve scaler for inference

8. **Class Imbalance Handling**
   - Analyze churn distribution
   - Apply SMOTE or class weighting (if required)

9. **Train–Test Split**
   - Stratified split (80% train / 20% test)
   - Preserve churn ratio

**Output:** Clean, numeric, balanced, model-ready dataset

## Feature Engineering Workflow

**Input:** Cleaned & Preprocessed Dataset

1. **Recency Feature Generation**
   - Compute days since last purchase
   - Create Days_Since_Last_Purchase

2. **Engagement Feature Generation**
   - Combine website visits & session time
   - Create Engagement_Score

3. **Purchase Behavior Features**
   - Calculate total spend
   - Calculate average spend per visit
   - Create purchase intensity indicators

4. **Loyalty Feature Construction**
   - Encode membership tiers
   - Incorporate referral count
   - Create loyalty indicators

5. **Support Risk Feature Creation**
   - Analyze support tickets
   - Combine with satisfaction score
   - Generate churn risk indicator

6. **Demographic Segmentation**
   - Bucket age into segments
   - Create age-group feature

7. **Feature Refinement**
   - Remove redundant raw attributes
   - Retain high-signal engineered features
   - Finalize feature list

**Output:** High-impact, business-driven, engineered feature set

## Why these models?

### Logistic Regression
- Works very well for Yes/No (binary) problems like churn
- Gives clear explanation of which features increase churn
- Fast to train and reliable as a baseline model

### Random Forest
- Captures non-linear patterns (real customer behavior is not linear)
- Combines many decision trees → more accurate
- Handles noisy and complex data better than simple models
- Gives feature importance (which factors matter most)

### XGBoost
- Focuses more on hard-to-predict churn cases
- Reduces both bias and error
- XGBoost achieved the best Recall and ROC-AUC, so we selected it as the final model

## Metric Scores

### Confusion Matrix
We start evaluation with the confusion matrix, which shows how many predictions are correct and where the model makes mistakes.

### Accuracy
Accuracy indicates overall correctness; however, due to class imbalance, it does not reflect the model's ability to identify churners reliably

### Recall
Recall measures the model's ability to correctly identify actual churners. A higher recall ensures that fewer churn-prone customers are missed.

### Precision
Precision measures how many predicted churners actually churned. Higher precision helps reduce unnecessary retention offers and marketing costs.

### ROC-AUC
The ROC curve shows how well the Random Forest separates churners from non-churners across different thresholds.

---

## Customer Churn Prediction Report: Online Retail Dataset

### 1. Use Case Description
The objective of this project is to build a machine learning model that predicts customer churn for an online retail business. Early identification of customers at risk of churning allows the company to implement targeted retention strategies (e.g., personalized offers, improved support), thereby reducing revenue loss.

This analysis is based on a larger, more comprehensive real-world dataset: `online_retail_customer_data_extended.csv` with ~9,000 customer records. Key evaluation metrics: **Accuracy, Precision, Recall, ROC-AUC, Confusion Matrix**. Given the class imbalance (~20% churn), **Recall** is particularly important for capturing at-risk customers.

### 2. Dataset Description
- Rows: 9,000
- Columns: 17 original features
- Target Variable: `Churn` (1 = Churned, 0 = Retained)
- Churn Distribution: 1,776 churned (19.73%), 7,224 retained (80.27%) → Significant class imbalance

Key Features (type grouped):
- Demographic: `Age`, `Gender`
- Financial: `Annual_Income_USD`, `Spending_Score`, `Total_Purchases`, `Avg_Purchase_Value`, `Total_Spend` (engineered)
- Behavioral: `Website_Visits_Last_Month`, `Avg_Time_Per_Visit_Minutes`, `Engagement_Score` (engineered), `Days_Since_Last_Purchase` (engineered)
- Support: `Support_Tickets_Last_6_Months`, `Satisfaction_Score`, `Support_Risk_Score` (engineered)
- Loyalty: `Membership_Status`, `Referred_Friends`, `Is_Loyal_Customer` (engineered)
- Categorical: `Preferred_Payment_Method`, `Region`

No missing values were present after initial checks.

### 3. Data Preprocessing & Feature Engineering
- Dropped irrelevant column: `CustomerID`
- Converted `Last_Purchase_Date` to datetime and engineered `Days_Since_Last_Purchase` (recency)
- Engineered additional predictive features:
  - `Total_Spend` = `Total_Purchases` × `Avg_Purchase_Value`
  - `Avg_Spend_Per_Visit`
  - `Engagement_Score` = `Website_Visits_Last_Month` × `Avg_Time_Per_Visit_Minutes`
  - `Is_Loyal_Customer` (based on referrals or premium membership)
  - `Support_Risk_Score` = `Support_Tickets_Last_6_Months` × (5 – `Satisfaction_Score`)
- One-hot encoded categorical variables: `Membership_Status`, `Preferred_Payment_Method`, `Region`
- Label encoded `Gender`
- Final feature count after engineering: **27 features**

### 4. Exploratory Data Analysis (Key Insights)
- Higher churn associated with:
  - More support tickets
  - Lower satisfaction scores
  - Longer time since last purchase
  - Lower engagement (visits × time)
- Certain payment methods and regions showed slightly elevated churn rates
- No strong linear correlations among numerical features, but engineered features (e.g., `Support_Risk_Score`) logically align with churn risk

### 5. Model Training & Handling Imbalance
- Train–test split: 80% train (7,200), 20% test (1,800), stratified
- Scaling: `StandardScaler` applied to numerical features
- Imbalance handling approaches used:
  - Logistic Regression with `class_weight='balanced'`
  - Random Forest with `class_weight` adjusted (example: `{0:1, 1:3}`)
  - XGBoost with `scale_pos_weight` tuned (≈ 4.06)
  - SMOTE oversampling applied to training data for exploration

Three models were trained and evaluated (sample metrics):

- Logistic Regression (threshold 0.5): Accuracy 0.508, Precision 0.199, Recall 0.493, ROC-AUC 0.502
- Random Forest (threshold 0.35): Accuracy 0.414, Precision 0.207, Recall 0.696, ROC-AUC 0.527
- XGBoost (threshold 0.5): Accuracy 0.693, Precision 0.225, Recall 0.228, ROC-AUC 0.530

**Threshold analysis (XGBoost)**: Lowering probability threshold (0.3–0.4) increases Recall (0.54–0.78) but reduces Accuracy; default 0.5 gives balanced accuracy with moderate Recall.

**ROC-style summary**: XGBoost achieved the highest ROC-AUC (~0.530), followed by Random Forest (~0.527). Overall discriminative power is moderate, indicating room for richer features.

### 6. Model Evaluation Summary
- **Best Accuracy:** XGBoost (~69.3%)
- **Best Recall (critical for churn detection):** Random Forest at threshold ≈ 0.35 (≈69.6% of churners identified)
- Trade-off: High recall sacrifices accuracy/precision due to imbalance
- Overall Predictive Power: Moderate (ROC-AUC ~0.53 across models)

**Business Recommendation:**
- Use Random Forest with threshold ≈ 0.35 if the goal is to maximize identification of at-risk customers (acceptable false positives for retention campaigns).
- Use XGBoost at default threshold if false positives are costly and a more conservative approach is preferred.

### 7. Conclusion & Recommendations
The models predict customer churn with moderate performance on this 9,000-record online retail dataset. Class imbalance (20% churn) was addressed through weighting and SMOTE, enabling reasonable recall rates.

**Key drivers of churn (from tree-model importance & EDA):**
- High support tickets + low satisfaction
- Low recent engagement and spending
- Longer recency since last purchase

**Next steps:**
- Collect additional behavioral features (e.g., product category preferences, discount usage)
- Experiment with ensembles and stacking (RF + XGBoost)
- Implement model in production with probability thresholds tuned to business cost of false positives vs. missed churn
- Monitor model drift as customer behavior evolves

---
*Added: Customer Churn Prediction Report (detailed summary, 2025-12-22)*