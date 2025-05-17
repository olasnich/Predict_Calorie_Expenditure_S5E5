# Predict_Calorie_Expenditure_S5E5
For Kaggle Playground series

# Option 1: Run ensemble with hyperparameter tuning
1. Create conda environment using env.yaml

```markdown
    conda env create -f env.yml
    conda activate myenv
```

2. Run train_model.py

```markdown
    python train_model.py
```

# Option 2: Run Hyperparameter tuning for LGBM, XGB and CatBoost

1. Run .bat file

```markdown
    run_python_scripts.bat <env name>
```    

2. Create submissions in create_submissions.ipynb