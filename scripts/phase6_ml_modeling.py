"""
Phase 6+7: ML Prediction & Model Evaluation
Air Quality Inspection & Prediction (Pan-India)

Trains models once, uses them for both prediction outputs and evaluation outputs.
No model persistence (no pickle/joblib) - everything runs in a single pass.

Input:  data/processed/city_day_features.csv
Outputs:
  - outputs/tables/predictions_table.csv
  - outputs/tables/model_metrics_table.csv
  - outputs/figures/feature_importance.png
  - outputs/figures/actual_vs_predicted.png
  - outputs/figures/model_comparison.png
  - outputs/figures/confusion_matrix.png
"""

import warnings
warnings.filterwarnings('ignore')

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import (r2_score, mean_absolute_error, mean_squared_error,
                              accuracy_score, precision_score, recall_score, f1_score,
                              confusion_matrix)

IN_PATH = '/home/nv/da379/data/processed/city_day_features.csv'
PRED_TABLE_OUT = '/home/nv/da379/outputs/tables/predictions_table.csv'
METRICS_TABLE_OUT = '/home/nv/da379/outputs/tables/model_metrics_table.csv'
FI_FIG = '/home/nv/da379/outputs/figures/feature_importance.png'
AVP_FIG = '/home/nv/da379/outputs/figures/actual_vs_predicted.png'
COMPARISON_FIG = '/home/nv/da379/outputs/figures/model_comparison.png'
CM_FIG = '/home/nv/da379/outputs/figures/confusion_matrix.png'

os.makedirs('/home/nv/da379/outputs/tables', exist_ok=True)
os.makedirs('/home/nv/da379/outputs/figures', exist_ok=True)

df = pd.read_csv(IN_PATH, parse_dates=['Date'])

feature_cols = ['PM2.5','PM10','NO','NO2','NOx','NH3','CO','SO2','O3','Benzene','Toluene',
                 'Year','Month','Day','DayOfWeek','City_enc','State_enc',
                 'AQI_lag1','AQI_lag7','AQI_roll7','AQI_roll30',
                 'Season_Monsoon','Season_Post-Monsoon','Season_Summer','Season_Winter']

X = df[feature_cols]
y_reg = df['AQI']
y_clf = df['AQI_Bucket']

X_train, X_test, y_reg_train, y_reg_test, y_clf_train, y_clf_test = train_test_split(
    X, y_reg, y_clf, test_size=0.2, random_state=42
)

# ================= REGRESSION: TRAIN + PREDICT =================
reg_models = {
    'LinearRegression': LinearRegression(),
    'DecisionTree': DecisionTreeRegressor(max_depth=10, random_state=42),
    'RandomForest': RandomForestRegressor(n_estimators=100, max_depth=12, random_state=42, n_jobs=-1)
}

reg_metrics_rows = []
reg_predictions = {}
for name, model in reg_models.items():
    model.fit(X_train, y_reg_train)
    preds = model.predict(X_test)
    reg_predictions[name] = preds
    reg_metrics_rows.append({
        'Task': 'Regression', 'Model': name,
        'R2': round(r2_score(y_reg_test, preds), 3),
        'MAE': round(mean_absolute_error(y_reg_test, preds), 2),
        'RMSE': round(np.sqrt(mean_squared_error(y_reg_test, preds)), 2),
        'Accuracy': None, 'Precision': None, 'Recall': None, 'F1': None
    })

best_reg_name = max(reg_models, key=lambda k: r2_score(y_reg_test, reg_predictions[k]))
print("=== REGRESSION RESULTS (predicting AQI) ===")
for row in reg_metrics_rows:
    print(f"{row['Model']}: R2={row['R2']}, MAE={row['MAE']}, RMSE={row['RMSE']}")
print(f"Best regression model: {best_reg_name}\n")

# ================= CLASSIFICATION: TRAIN + PREDICT =================
clf_models = {
    'LogisticRegression': LogisticRegression(max_iter=1000, random_state=42),
    'DecisionTree': DecisionTreeClassifier(max_depth=10, random_state=42),
    'RandomForest': RandomForestClassifier(n_estimators=100, max_depth=12, random_state=42, n_jobs=-1)
}

clf_metrics_rows = []
clf_predictions = {}
for name, model in clf_models.items():
    model.fit(X_train, y_clf_train)
    preds = model.predict(X_test)
    clf_predictions[name] = preds
    clf_metrics_rows.append({
        'Task': 'Classification', 'Model': name,
        'R2': None, 'MAE': None, 'RMSE': None,
        'Accuracy': round(accuracy_score(y_clf_test, preds), 3),
        'Precision': round(precision_score(y_clf_test, preds, average='weighted', zero_division=0), 3),
        'Recall': round(recall_score(y_clf_test, preds, average='weighted', zero_division=0), 3),
        'F1': round(f1_score(y_clf_test, preds, average='weighted', zero_division=0), 3)
    })

best_clf_name = max(clf_models, key=lambda k: accuracy_score(y_clf_test, clf_predictions[k]))
print("=== CLASSIFICATION RESULTS (predicting AQI_Bucket) ===")
for row in clf_metrics_rows:
    print(f"{row['Model']}: Accuracy={row['Accuracy']}, F1={row['F1']}")
print(f"Best classification model: {best_clf_name}\n")

# ---------- Metrics table (Phase 7 output) ----------
metrics_df = pd.DataFrame(reg_metrics_rows + clf_metrics_rows)
metrics_df.to_csv(METRICS_TABLE_OUT, index=False)

# ---------- Predictions table (Phase 6 output) ----------
pred_table = X_test.copy()
pred_table['City'] = df.loc[X_test.index, 'City'].values
pred_table['Date'] = df.loc[X_test.index, 'Date'].values
pred_table['Actual_AQI'] = y_reg_test.values
pred_table['Predicted_AQI'] = reg_predictions[best_reg_name]
pred_table['Actual_Bucket'] = y_clf_test.values
pred_table['Predicted_Bucket'] = clf_predictions[best_clf_name]
pred_table = pred_table[['City','Date','Actual_AQI','Predicted_AQI','Actual_Bucket','Predicted_Bucket']]
pred_table.to_csv(PRED_TABLE_OUT, index=False)

# ---------- Feature importance (Phase 6 output) ----------
rf_reg = reg_models['RandomForest']
importances = pd.Series(rf_reg.feature_importances_, index=feature_cols).sort_values(ascending=True)

plt.figure(figsize=(10,8))
importances.plot(kind='barh', color='teal')
plt.title('Feature Importance - Random Forest (AQI Regression)')
plt.xlabel('Importance')
plt.tight_layout()
plt.savefig(FI_FIG, dpi=150)
plt.close()

# ---------- Actual vs Predicted (Phase 6 output) ----------
plt.figure(figsize=(8,8))
plt.scatter(y_reg_test, reg_predictions[best_reg_name], alpha=0.3, s=10, color='steelblue')
lims = [0, max(y_reg_test.max(), reg_predictions[best_reg_name].max())]
plt.plot(lims, lims, 'r--', label='Perfect Prediction')
plt.xlabel('Actual AQI')
plt.ylabel('Predicted AQI')
plt.title(f'Actual vs Predicted AQI ({best_reg_name})')
plt.legend()
plt.tight_layout()
plt.savefig(AVP_FIG, dpi=150)
plt.close()

# ---------- Model comparison bar chart (Phase 7 output) ----------
fig, axes = plt.subplots(1, 2, figsize=(14,5))
reg_names = [r['Model'] for r in reg_metrics_rows]
reg_r2 = [r['R2'] for r in reg_metrics_rows]
axes[0].bar(reg_names, reg_r2, color='steelblue')
axes[0].set_title('Regression Model Comparison (R2 Score)')
axes[0].set_ylabel('R2 Score')
axes[0].set_ylim(0,1)

clf_names = [r['Model'] for r in clf_metrics_rows]
clf_acc = [r['Accuracy'] for r in clf_metrics_rows]
axes[1].bar(clf_names, clf_acc, color='darkorange')
axes[1].set_title('Classification Model Comparison (Accuracy)')
axes[1].set_ylabel('Accuracy')
axes[1].set_ylim(0,1)

plt.tight_layout()
plt.savefig(COMPARISON_FIG, dpi=150)
plt.close()

# ---------- Confusion matrix (best classifier) (Phase 7 output) ----------
bucket_order = ['Good','Satisfactory','Moderate','Poor','Very Poor','Severe']
cm = confusion_matrix(y_clf_test, clf_predictions[best_clf_name], labels=bucket_order)

plt.figure(figsize=(8,7))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=bucket_order, yticklabels=bucket_order)
plt.title(f'Confusion Matrix - {best_clf_name} (AQI_Bucket)')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.tight_layout()
plt.savefig(CM_FIG, dpi=150)
plt.close()

print(f"Predictions table saved to {PRED_TABLE_OUT}")
print(f"Metrics table saved to {METRICS_TABLE_OUT}")
print(f"Feature importance plot saved to {FI_FIG}")
print(f"Actual vs predicted plot saved to {AVP_FIG}")
print(f"Model comparison plot saved to {COMPARISON_FIG}")
print(f"Confusion matrix saved to {CM_FIG}")
