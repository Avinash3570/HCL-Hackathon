# HCL-Hackathon

## HCL Tech Hackathon

**Git Repo** - https://github.com/Avinash3570/HCL-Hackathon.git

**Dataset** - https://www.kaggle.com/datasets/sahilislam007/online-retail-customer-churn-prediction-dataset

## Procedure:

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