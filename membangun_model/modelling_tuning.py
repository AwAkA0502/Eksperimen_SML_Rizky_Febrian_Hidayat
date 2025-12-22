import os
os.environ["QT_QPA_PLATFORM"] = "offscreen"
import pandas as pd
import numpy as np
import mlflow
import mlflow.catboost
import matplotlib.pyplot as plt
import seaborn as sns
import dagshub
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from catboost import CatBoostRegressor
import warnings
import json
from datetime import datetime
warnings.filterwarnings('ignore')

print("="*80)
print("MODELING - ADVANCED (CATBOOST + TUNING + DAGSHUB)")
print("="*80)

dagshub.init(
    repo_owner="AwAkA0502",
    repo_name="Eksperimen_SML_Rizky_Febrian_Hidayat",
    mlflow=True
)

df = pd.read_csv('./preprocessing/bmw_preprocessing/bmw_preprocessing.csv')

feature_columns = ['year', 'mileage', 'tax', 'mpg', 'engineSize',
                   'model_encoded', 'transmission_encoded', 'fuelType_encoded']
X = df[feature_columns]
y = df['price']

X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.15, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.176, random_state=42)

mlflow.set_experiment("BMW_Price_Prediction_Advanced")

param_grid = {
    'iterations': [100, 200],
    'depth': [4, 6],
    'learning_rate': [0.05, 0.1],
    'l2_leaf_reg': [1, 3]
}

base_model = CatBoostRegressor(loss_function='RMSE', verbose=False, random_seed=42)
grid_search = GridSearchCV(estimator=base_model, param_grid=param_grid, cv=3, scoring='r2', n_jobs=-1)
grid_search.fit(X_train, y_train)
best_params = grid_search.best_params_

with mlflow.start_run(run_name=f"CatBoost_Final_Run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
    for param, value in best_params.items():
        mlflow.log_param(param, value)
    
    final_model = CatBoostRegressor(**best_params, loss_function='RMSE', verbose=False, random_seed=42)
    final_model.fit(X_train, y_train, eval_set=(X_val, y_val))
    
    y_pred = final_model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    mlflow.log_metric("rmse", rmse)
    mlflow.log_metric("mae", mae)
    mlflow.log_metric("r2_score", r2)
    
    importances = final_model.get_feature_importance()
    importance_df = pd.DataFrame({'Feature': feature_columns, 'Importance': importances}).sort_values(by='Importance', ascending=False)
    importance_df.to_csv('feature_importance.csv', index=False)
    mlflow.log_artifact("feature_importance.csv")
    
    model_summary = {
        "model_name": "CatBoost Regressor (Tuned)",
        "best_parameters": best_params,
        "performance_metrics": {"r2_score": float(r2), "mae": float(mae), "rmse": float(rmse)},
        "features_used": feature_columns,
        "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    with open('model_summary.json', 'w') as f:
        json.dump(model_summary, f, indent=4)
    mlflow.log_artifact("model_summary.json")
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    axes[0].scatter(y_test, y_pred, alpha=0.5)
    axes[0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    axes[0].set_title('Actual vs Predicted')
    
    pd.Series(importances, index=feature_columns).sort_values().plot(kind='barh', ax=axes[1])
    axes[1].set_title('Feature Importance Plot')
    
    plt.tight_layout()
    plt.savefig("adv_eval_plot.png")
    mlflow.log_artifact("adv_eval_plot.png")
    
    mlflow.catboost.log_model(final_model, "catboost_model")
    print(f"DONE! R2: {r2:.4f}")