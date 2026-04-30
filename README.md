# SECOM Semiconductor Defect Detection

## Dataset
[UCI SECOM](https://www.kaggle.com/datasets/paresh2047/uci-semcom) — 1567 samples, 590 sensor features, about 6.6% failure rate

## EDA Findings
- The raw file includes a timestamp column plus 16 all-null string columns.
- Those string columns were dropped during cleaning.
- Average missing values per row: 15.55.
- 104 duplicated columns were removed.
- Labels are encoded as `-1` / `1` and were remapped to `0` / `1` for classification.

## Modeling Approach
- Mean imputation for missing values
- `VarianceThreshold` for constant features
- `SelectKBest(f_classif)` for feature selection
- `StandardScaler`
- `XGBClassifier(scale_pos_weight=14, booster='gbtree', eta=0.4, max_depth=10)`
- Decision-threshold tuning on validation probabilities

## Results
- Best grid-search F1: `0.180`
- Best threshold: `0.236`
- Test accuracy: `0.93`
- Minority class F1: `0.36`
- Minority class recall: `0.29`

## What I Tried
- XGBoost vs RandomForest — XGBoost won
- Grid search over `k`, tree depth, and learning rate
- Threshold tuning improved minority-class performance

## Serving
- FastAPI app in `serving/main.py`
- Loads `serving/finalpipe.joblib`
- `/predict` expects 590 raw features
- Requires `SECOM_API_KEY`
