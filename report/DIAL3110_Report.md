# Loan Prediction using Machine Learning

**DIAL3110 Fundamentals of Artificial Intelligence**

**Group members:** (add 3 names + roles)

---

## Abstract
(Write 120-180 words: problem, dataset of 614 applications, three models compared (LR, RF, GB), Logistic Regression selected with F1 0.9081, deployed Streamlit dashboard for real-time and batch prediction.)

---

## 1. Introduction
### 1.1 Problem statement
Motz Financial Services must decide whether each loan applicant is approved (Y) or rejected (N) using demographics, income, loan details, and credit history. Manual screening is slow, subjective, and inconsistent across reviewers.

### 1.2 Relevance and importance
An automated, reproducible prediction system reduces review time, ensures consistent decisions, and supports risk-aware lending. It also allows batch scoring of many applications at once.

### 1.3 Objectives and scope
Objectives: (1) build a reproducible ML pipeline; (2) compare Logistic Regression, Random Forest and Gradient Boosting; (3) deploy a live dashboard for single and batch predictions; (4) evaluate on held-out data.

Scope: 614 loan applications, 13 columns, binary classification. Excludes time-series or macroeconomic features.

---

## 2. Body

### 2.1 Data gathering and cleaning (Task 1)

- **Missing values:** handled in `src/data_cleaning.py` using median imputation for numeric features, mode for categoricals, and mode specifically for `Credit_History` (which is strongly predictive and cannot be safely defaulted to a constant).
- **Outliers:** flagged via the IQR method in `src/data_cleaning.py`. High-income and high-loan outliers were retained because they are genuine applicant profiles, not measurement errors.
- **Descriptive analysis:** target balance (`figures/fig1_target.png`), income vs loan amount (`fig2_income_loan.png`), outlier distributions (`fig3_outliers.png`), credit history breakdown (`fig4_credit.png`), and correlation matrix (`fig5_corr.png`). The target is imbalanced (~69% approved / 31% rejected), motivating F1 and ROC-AUC as primary metrics rather than accuracy alone.

### 2.2 Data preprocessing (Task 2)

- **Feature extraction (`engineer_features` in `src/preprocessing.py`):**
  - `TotalIncome` = ApplicantIncome + CoapplicantIncome
  - `LoanAmount_log` = log(1 + LoanAmount) to reduce skew
  - `EMI` = LoanAmount / Loan_Amount_Term
  - `Balance_Income` = TotalIncome - EMI
- **Feature tuning:** categorical features one-hot encoded; numeric features standard-scaled for Logistic Regression. All transformations are wrapped in a `ColumnTransformer` inside a scikit-learn `Pipeline`, ensuring the transformer is fit only on training folds during cross-validation (no leakage).

### 2.3 Model determination (Task 3)

Three classifiers with complementary bias-variance profiles were selected:

- **Logistic Regression** - interpretable linear baseline. Coefficients map directly to log-odds, valuable in a regulated lending context where decisions must be explainable.
- **Random Forest** - bagged ensemble of decision trees. Captures non-linear interactions and is robust to outliers and unscaled features.
- **Gradient Boosting** - sequentially boosted ensemble. Typically achieves the highest accuracy on tabular data by correcting residual errors of prior trees.

The rationale was to compare a transparent linear model against two non-linear ensembles, so any accuracy gain could be weighed against interpretability. All three share the same preprocessing pipeline, ensuring a fair comparison.

### 2.4 Model training (Task 4)

`train_pipeline.py` orchestrates the full training workflow:

1. Load `data/loan_data_set.csv` and drop `Loan_ID` and `Loan_Status`.
2. Apply `engineer_features` to create `TotalIncome`, `LoanAmount_log`, `EMI`, `Balance_Income`.
3. Build a scikit-learn `Pipeline` per model: `ColumnTransformer` (OneHotEncoder + StandardScaler) + estimator. Preprocessing inside the pipeline prevents leakage because the transformer is fit only on the training fold during CV.
4. Stratified 80/20 train-test split (`random_state=42`) to preserve the ~69/31 class balance.
5. `GridSearchCV` with 5-fold CV on the training set only, scoring by F1 (appropriate for the mildly imbalanced target).
6. Serialise each tuned pipeline with `joblib` to `models/`.

### 2.5 Model tuning and results (Task 5)

Hyperparameter tuning used exhaustive `GridSearchCV` (5-fold, F1 scoring). Grids explored regularisation and complexity controls: `C` for Logistic Regression; `n_estimators`, `max_depth`, `min_samples_leaf` for Random Forest; `n_estimators`, `learning_rate`, `max_depth` for Gradient Boosting. Each best estimator was refit on the full training set and evaluated once on the held-out test set (`src/evaluate.py`).

**Table 1. Held-out test-set performance** (`models/model_comparison.csv`):

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression | 0.8618 | 0.8400 | 0.9882 | **0.9081** | **0.8437** |
| Random Forest | 0.7967 | 0.8659 | 0.8353 | 0.8503 | 0.8310 |
| Gradient Boosting | 0.8293 | 0.8265 | 0.9529 | 0.8852 | 0.8065 |

**Model selection.** `train_pipeline.py` selects the final model by highest F1 on the held-out test set (`results_df["F1"].idxmax()`), which chose **Logistic Regression** (F1 = 0.9081). This is stated explicitly because it is a point of methodological scrutiny: selecting by test-set F1 rather than cross-validated F1 can slightly overstate generalisation. For a lending use-case, the high recall of Logistic Regression (0.9882) is arguably more valuable than raw accuracy, because it minimises false rejections of creditworthy applicants.

The result that a linear model outperformed two ensembles is notable: with only 614 rows, the engineered features (`TotalIncome`, `EMI`, `Balance_Income`, `LoanAmount_log`) already capture the dominant signal, and the tree ensembles are more prone to overfitting on small samples.

### 2.6 Prediction dashboard (Task 6)

The dashboard (`app.py`) is deployed at:

https://loan-prediction-system-ondttmjtbnxjjptr4bg2gm.streamlit.app/

It performs **inference only** - no training occurs on the cloud. The flow is:

1. `joblib.load("models/best_model.joblib")` loads the serialised pipeline (preprocessor + estimator).
2. For single predictions, user inputs are assembled into a one-row DataFrame, passed through `engineer_features`, then `model.predict` / `model.predict_proba`.
3. For batch predictions, an uploaded CSV is cleaned, passed through `engineer_features`, then predicted; results are downloadable as `loan_predictions.csv`.

This satisfies the assignment requirement that the model be testable on unseen out-of-sample data: any CSV with the same raw columns as the training set flows through the identical feature-engineering and preprocessing chain before prediction. A sample of 10 held-out predictions with probabilities is included in `out_of_sample_proof.txt`.

---

## 3. Conclusion
(Write one paragraph: Logistic Regression achieved the best F1 (0.9081) and ROC-AUC (0.8437); a live Streamlit dashboard supports both single and batch prediction; limitations include small dataset and test-set model selection; future work includes class-weight tuning, additional models (XGBoost), and larger datasets.)

---

## References
(IEEE style, 20+ references. To be added.)