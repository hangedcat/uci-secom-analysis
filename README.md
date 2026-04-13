# SECOM Semiconductor Defect Detection

## Dataset
UCI SECOM — 1567 samples, 590 features, 6.6% failure rate

## Approach
- Missing value imputation (mean)
- Variance threshold + SelectKBest (k=7) feature selection  
- RandomForestClassifier with balanced class weights

## Results
- CV F1: 0.14
- Test F1: 0.29
- Minority class recall: 0.38

## What I tried
- XGBoost vs RandomForest — RF won
- mutual_info_classif vs f_classif — f_classif won
- k values from 3 to 200 — k=7 optimal
- Various n_estimators and max_depth combinations

## What didn't work
- mutual_info_classif degraded performance
- Increasing model complexity beyond n_estimators=300, max_depth=5 showed no improvement