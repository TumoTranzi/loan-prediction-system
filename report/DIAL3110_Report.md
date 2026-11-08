# Loan Prediction using Machine Learning

**DIAL3110 Fundamentals of Artificial Intelligence**

# Team

| Member | Responsibility |
|---|---|
| Member 1 | Data gathering, cleaning, descriptive analysis (Task 1) |
| Member 2 | Preprocessing, model training and tuning (Tasks 2, 4, 5) |
| Member 3 | Dashboard, report and presentation (Tasks 6, 8) |

---

## Abstract
This report presents a machine learning system for predicting loan approval at Motz Financial Services. Using a public dataset of 614 loan applications with 13 applicant, income, loan and credit features, we implement a reproducible pipeline covering data cleaning (median/mode imputation, IQR outlier flagging), feature engineering (TotalIncome, LoanAmount_log, EMI, Balance_Income), and preprocessing (one-hot encoding, standard scaling). Three classifiers - Logistic Regression, Random Forest and Gradient Boosting - were trained with stratified 80/20 splits and tuned using 5-fold GridSearchCV scored by F1. On the held-out test set, Logistic Regression achieved the best F1 (0.9081) and ROC-AUC (0.8437), with recall of 0.9882 - an attractive property for lending, where false rejections of creditworthy applicants are costly. The tuned pipeline is deployed as a live Streamlit dashboard supporting both single and batch CSV prediction, with no retraining required on the cloud.

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
Logistic Regression achieved the strongest overall performance on the held-out test set (F1 = 0.9081, ROC-AUC = 0.8437, recall = 0.9882), outperforming Random Forest and Gradient Boosting on this small 614-row dataset, where the engineered features already capture most of the predictive signal and tree ensembles are more prone to overfitting. The trained pipeline is deployed as a live Streamlit dashboard at https://loan-prediction-system-ondttmjtbnxjjptr4bg2gm.streamlit.app/, providing both interactive single-applicant prediction and batch CSV scoring - the same interface can be used to evaluate the formatted out-of-sample data on presentation day. Limitations include the modest dataset size, a model-selection rule that uses test-set F1 which can slightly overstate generalisation, and the absence of calibrated probability thresholds for regulatory use. Future work includes class-weight tuning and SMOTE for imbalance, adding XGBoost or LightGBM for comparison, calibration curves, and validation on larger, more recent loan portfolios.

---

## References
## References

[1] F. Pedregosa et al., Scikit-learn: Machine learning in Python, Journal of Machine Learning Research, vol. 12, pp. 2825-2830, 2011.
[2] L. Breiman, Random forests, Machine Learning, vol. 45, no. 1, pp. 5-32, 2001.
[3] J. H. Friedman, Greedy function approximation: A gradient boosting machine, Annals of Statistics, vol. 29, no. 5, pp. 1189-1232, 2001.
[4] D. W. Hosmer, S. Lemeshow, and R. X. Sturdivant, Applied Logistic Regression, 3rd ed. Hoboken, NJ: Wiley, 2013.
[5] T. Hastie, R. Tibshirani, and J. Friedman, The Elements of Statistical Learning, 2nd ed. New York: Springer, 2009.
[6] G. James, D. Witten, T. Hastie, and R. Tibshirani, An Introduction to Statistical Learning. New York: Springer, 2013.
[7] T. Fawcett, An introduction to ROC analysis, Pattern Recognition Letters, vol. 27, no. 8, pp. 861-874, 2006.
[8] D. M. Powers, Evaluation: From precision, recall and F-measure to ROC, informedness, markedness and correlation, Journal of Machine Learning Technologies, vol. 2, no. 1, pp. 37-63, 2011.
[9] N. V. Chawla, K. W. Bowyer, L. O. Hall, and W. P. Kegelmeyer, SMOTE: Synthetic minority over-sampling technique, Journal of Artificial Intelligence Research, vol. 16, pp. 321-357, 2002.
[10] C. M. Bishop, Pattern Recognition and Machine Learning. New York: Springer, 2006.
[11] T. Chen and C. Guestrin, XGBoost: A scalable tree boosting system, in Proc. 22nd ACM SIGKDD, 2016, pp. 785-794.
[12] G. Ke et al., LightGBM: A highly efficient gradient boosting decision tree, in Advances in Neural Information Processing Systems, vol. 30, 2017.
[13] J. Brownlee, Imbalanced Classification with Python. Melbourne: Machine Learning Mastery, 2020.
[14] A. Geron, Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow, 2nd ed. Sebastopol, CA: O Reilly, 2019.
[15] S. Raschka and V. Mirjalili, Python Machine Learning, 3rd ed. Birmingham: Packt, 2019.
[16] Streamlit Inc., Streamlit documentation, 2024. [Online]. Available: https://docs.streamlit.io
[17] Streamlit Inc., Streamlit Community Cloud, 2024. [Online]. Available: https://docs.streamlit.io/streamlit-community-cloud
[18] Joblib Developers, Joblib: Running Python functions as pipeline jobs, 2024. [Online]. Available: https://joblib.readthedocs.io
[19] Pandas Development Team, pandas documentation, 2024. [Online]. Available: https://pandas.pydata.org/docs/
[20] NumPy Developers, NumPy documentation, 2024. [Online]. Available: https://numpy.org/doc/
[21] scikit-learn Developers, Pipeline and ColumnTransformer, 2024. [Online]. Available: https://scikit-learn.org/stable/modules/compose.html
[22] Kaggle, Loan prediction dataset, 2024. [Online]. Available: https://www.kaggle.com/datasets/altruistdelhite04/loan-prediction-problem-dataset
[23] Y. Freund and R. E. Schapire, A decision-theoretic generalization of on-line learning and an application to boosting, Journal of Computer and System Sciences, vol. 55, no. 1, pp. 119-139, 1997.
[24] L. Breiman, J. Friedman, C. J. Stone, and R. A. Olshen, Classification and Regression Trees. Boca Raton, FL: CRC Press, 1984.
[25] S. B. Kotsiantis, Supervised machine learning: A review of classification techniques, Informatica, vol. 31, pp. 249-268, 2007.