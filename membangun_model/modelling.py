import os
os.environ["QT_QPA_PLATFORM"] = "offscreen"
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from catboost import CatBoostRegressor
import mlflow
import mlflow.catboost
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("MODELING - BASIC (MLFLOW AUTOLOG - CATBOOST)")
print("="*80)

print("\n[1/7] Loading preprocessed data...")
df = pd.read_csv('./preprocessing/bmw_preprocessing/bmw_preprocessing.csv')
print(f"✓ Data loaded: {len(df)} rows, {df.shape[1]} columns")

print("\n[2/7] Preparing features and target...")
feature_columns = ['year', 'mileage', 'tax', 'mpg', 'engineSize',
                   'model_encoded', 'transmission_encoded', 'fuelType_encoded']

X = df[feature_columns]
y = df['price']

print(f"✓ Features shape: {X.shape}")
print(f"✓ Target range: ${y.min():,.0f} - ${y.max():,.0f}")

print("\n[3/7] Splitting data...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("\n[4/7] Setting up MLflow...")
mlflow.set_experiment("BMW_Price_Prediction_CatBoost")
mlflow.catboost.autolog()

print("\n[5/7] Training CatBoost model...")
with mlflow.start_run(run_name="CatBoost_Basic"):
    params = {
        'iterations': 500,
        'depth': 6,
        'learning_rate': 0.1,
        'random_seed': 42,
        'verbose': False,
        'loss_function': 'RMSE'
    }
    
    model = CatBoostRegressor(**params)
    model.fit(X_train, y_train, eval_set=(X_test, y_test))
    
    print("\n[6/7] Evaluating model on test set...")
    y_pred = model.predict(X_test)
    
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
    accuracy = r2 * 100
    
    print("\n" + "="*80)
    print(f"RMSE: ${rmse:,.2f}")
    print(f"MAE:  ${mae:,.2f}")
    print(f"R2:    {r2:.4f}")
    print(f"Acc:   {accuracy:.2f}%")
    print("="*80)
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('CatBoost Model Evaluation', fontsize=16, fontweight='bold')
    
    axes[0, 0].scatter(y_test, y_pred, alpha=0.5)
    axes[0, 0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    axes[0, 0].set_title('Actual vs Predicted')
    
    residuals = y_test - y_pred
    axes[0, 1].scatter(y_pred, residuals, alpha=0.5)
    axes[0, 1].axhline(y=0, color='r', linestyle='--')
    axes[0, 1].set_title('Residual Plot')
    
    axes[1, 0].hist(residuals, bins=50)
    axes[1, 0].set_title('Error Distribution')
    
    feat_importances = pd.Series(model.get_feature_importance(), index=X.columns).sort_values()
    feat_importances.plot(kind='barh', ax=axes[1, 1])
    axes[1, 1].set_title('Feature Importance')
    
    plt.tight_layout()
    plt.savefig('catboost_evaluation.png')
    plt.close()

print("\n" + "="*80)
print("✓ MODELING CATBOOST SELESAI!")
print("="*80)