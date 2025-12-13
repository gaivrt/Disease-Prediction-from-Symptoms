import optuna
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
import numpy as np
import pandas as pd
import sklearn.metrics as metrics
from sklearn.model_selection import StratifiedKFold
import os
import joblib
import sys
import matplotlib.pyplot as plt

# Ensure imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from src.data.loader import DataLoader

def objective(trial, model_name, X, y, encoder):
    # Stratified K-Fold for Robustness
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    scores = []
    
    param = {}
    
    if model_name == 'xgboost':
        param = {
            'verbosity': 0,
            'objective': 'multi:softmax',
            'device': 'cuda', # Enable GPU acceleration
            'num_class': len(encoder.classes_),
            'n_estimators': trial.suggest_int('n_estimators', 100, 500),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.3, log=True),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 1.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 1.0, log=True)
        }
        model = xgb.XGBClassifier(**param)
        
    elif model_name == 'lightgbm':
        param = {
            'objective': 'multiclass',
            'metric': 'multi_logloss',
            # 'device': 'gpu', # Start with CPU for LightGBM as PIP install often lacks GPU support. Uncomment if compiled with GPU.
            # XGBoost and CatBoost are usually safe.
            'verbosity': -1,
            'num_class': len(encoder.classes_),
            'n_estimators': trial.suggest_int('n_estimators', 100, 500),
            'num_leaves': trial.suggest_int('num_leaves', 20, 100),
            'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.3, log=True),
            'feature_fraction': trial.suggest_float('feature_fraction', 0.5, 1.0),
            'bagging_fraction': trial.suggest_float('bagging_fraction', 0.5, 1.0),
            'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
        }
        model = lgb.LGBMClassifier(**param)
        
    elif model_name == 'catboost':
        param = {
            'task_type': 'GPU', # Enable GPU acceleration
            'iterations': trial.suggest_int('iterations', 100, 500),
            'depth': trial.suggest_int('depth', 4, 10),
            'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.3, log=True),
            'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1e-8, 10.0, log=True),
            'loss_function': 'MultiClass',
            'logging_level': 'Silent'
        }
        model = cb.CatBoostClassifier(**param)

    # CV Loop
    for train_index, val_index in skf.split(X, y):
        X_tr, X_val = X.iloc[train_index], X.iloc[val_index]
        y_tr, y_val = y[train_index], y[val_index]
        
        model.fit(X_tr, y_tr)
        preds = model.predict(X_val)
        score = metrics.accuracy_score(y_val, preds)
        scores.append(score)
        
    return np.mean(scores)

def run_tuning(n_trials=20): # Short runs for demo, usually higher
    # Load Data
    loader = DataLoader()
    X_train, y_train, _, _, encoder = loader.load_data()
    
    models = ['xgboost', 'lightgbm', 'catboost']
    best_results = {}
    
    os.makedirs('./saved_model', exist_ok=True)
    os.makedirs('./output', exist_ok=True)
    
    for m in models:
        print(f"\n--- Tuning {m.upper()} ---")
        study = optuna.create_study(direction='maximize')
        study.optimize(lambda trial: objective(trial, m, X_train, y_train, encoder), n_trials=n_trials)
        
        print(f"Best params for {m}: {study.best_params}")
        print(f"Best CV Accuracy: {study.best_value:.4f}")
        
        # Train final model with best params
        if m == 'xgboost':
            final_model = xgb.XGBClassifier(**study.best_params)
        elif m == 'lightgbm':
            final_model = lgb.LGBMClassifier(**study.best_params)
        elif m == 'catboost':
            final_model = cb.CatBoostClassifier(**study.best_params)
            
        final_model.fit(X_train, y_train)
        
        # Save Model
        save_path = f'./saved_model/{m}_best.json'
        # joblib usually safer for wrappers
        joblib.dump(final_model, f'./saved_model/{m}_best.joblib')
        
        best_results[m] = study.best_value
        
        # Plot Optimization History (Simple Plot via Optuna)
        try:
            fig = optuna.visualization.matplotlib.plot_optimization_history(study)
            plt.title(f'{m} Optimization History')
            plt.tight_layout()
            plt.savefig(f'./output/tuning_history_{m}.png')
            plt.close()
        except Exception as e:
            print(f"Could not plot history for {m}: {e}")

    print("\n--- Final Leaderboard (CV Accuracy) ---")
    for m, score in best_results.items():
        print(f"{m}: {score:.4f}")

if __name__ == "__main__":
    run_tuning(n_trials=10) # Quick run to verify functionality
