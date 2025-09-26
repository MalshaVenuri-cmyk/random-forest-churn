# Customer Churn Prediction — Random Forest Classifier

Implemented a Random Forest classifier to predict customer churn.

## Steps
- Preprocessed churn dataset (dropped collinear charge columns).
- Built Random Forest pipeline (OHE for categorical + passthrough numerics).
- Cross-validated base model (Mean F1 ≈ 0.63, Mean AUC ≈ 0.91).
- Tuned hyperparameters with GridSearchCV (Best F1 ≈ 0.77).
- Evaluated on test set:
  - Accuracy: 0.939
  - Precision: 0.865
  - Recall: 0.674
  - F1: 0.757
  - AUC: 0.925
- Analyzed feature importance and threshold trade-offs.

## Tools
- Python
- Pandas, NumPy
- Scikit-learn
- Matplotlib
