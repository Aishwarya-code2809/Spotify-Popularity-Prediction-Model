import pandas as pd
import numpy as np
from google.colab import files
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import sys
import io
import time
print("--- 1/10: File Upload and Initial Data Loading ---")
print("Please select your 'dataset.csv' file now.")
# --- File Upload and Loading ---
try:
# Trigger the interactive file upload dialog
uploaded_files = files.upload()
uploaded_file_name = list(uploaded_files.keys())[0]
# Read the file
global df_final
df_final = pd.read_csv(io.BytesIO(uploaded_files[uploaded_file_name]))
print(f"\n✅ File '{uploaded_file_name}' loaded successfully. Initial shape: {df_final.shape}")
except Exception as e:
print(f"🛑 CRITICAL ERROR during file loading: {e}")
sys.exit(1)
if 'df_final' not in locals():
print("🛑 ERROR: df_final is not defined. Please run Block 1 first.")
else:
print("\n--- 2/8: Preprocessing Setup and Execution ---")
ACTUAL_GENRE_COLUMN_NAME = 'track_genre'
features_to_drop = ['Unnamed: 0', 'track_id', 'artists', 'album_name', 'track_name']
# Separate Target (y) and Features (X)
y = df_final['popularity']
X = df_final.drop(columns=['popularity'] + features_to_drop, errors='ignore')
# Feature lists
numerical_features = ['danceability', 'energy', 'loudness', 'speechiness',
'acousticness', 'instrumentalness', 'liveness', 'valence',
'tempo', 'duration_ms']
genre_feature = [ACTUAL_GENRE_COLUMN_NAME]
categorical_features = ['mode', 'time_signature', 'explicit']
# Create Preprocessing Pipeline
global preprocessor
preprocessor = ColumnTransformer(
transformers=[
('num', StandardScaler(), numerical_features),
('genre_ohe', OneHotEncoder(handle_unknown='ignore', sparse_output=False), genre_feature),
('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
],
remainder='passthrough'
)
X_processed = preprocessor.fit_transform(X)
global feature_names
feature_names = preprocessor.get_feature_names_out()
global X_processed_df
X_processed_df = pd.DataFrame(X_processed, columns=feature_names)
print(" Preprocessing complete. Data scaled and encoded.")
if 'df_final' not in locals():
print("🛑 ERROR: df_final is not defined. Please run Block 1 first.")
else:
print("\n--- 3/10: Feature and Target Separation ---")
ACTUAL_GENRE_COLUMN_NAME = 'track_genre'
features_to_drop = ['Unnamed: 0', 'track_id', 'artists', 'album_name', 'track_name']
# Separate Target (y) and Features (X)
global y, X
y = df_final['popularity']
X = df_final.drop(columns=['popularity'] + features_to_drop, errors='ignore')
# Feature lists
global numerical_features, categorical_features, genre_feature
numerical_features = ['danceability', 'energy', 'loudness', 'speechiness',
'acousticness', 'instrumentalness', 'liveness', 'valence',
'tempo', 'duration_ms']
genre_feature = [ACTUAL_GENRE_COLUMN_NAME]
categorical_features = ['mode', 'time_signature', 'explicit']
print(f" Target variable 'popularity' separated. X shape: {X.shape}")
if 'X' not in locals():
print("🛑 ERROR: X is not defined. Please run Block 3 first.")
else:
print("\n--- 4/10: Preprocessing Pipeline Setup and Fit ---")
# Create Preprocessing Pipeline
global preprocessor
preprocessor = ColumnTransformer(
transformers=[
# 1. StandardScaler for numerical features (scaling)
('num', StandardScaler(), numerical_features),
# 2. OneHotEncoder for the genre (encoding)
('genre_ohe', OneHotEncoder(handle_unknown='ignore', sparse_output=False), genre_feature),
# 3. OneHotEncoder for other categorical features
('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
],
remainder='passthrough'
)
global X_processed
X_processed = preprocessor.fit_transform(X)
print(" Preprocessing pipeline created and fitted.")
if 'X_processed' not in locals():
print("🛑 ERROR: X_processed is not defined. Please run Block 4 first.")
else:
print("\n--- 5/10: Convert to DataFrame and Train/Test Split ---")
# Convert processed data back to a DataFrame with proper column names
global feature_names
feature_names = preprocessor.get_feature_names_out()
global X_processed_df
X_processed_df = pd.DataFrame(X_processed, columns=feature_names)
# Perform the final data split
global X_train, X_test, y_train, y_test
X_train, X_test, y_train, y_test = train_test_split(
X_processed_df, y, test_size=0.2, random_state=42
)
print(f" Data split complete. Training set shape: {X_train.shape}")
from sklearn.metrics import mean_squared_error
if 'X_train' not in locals():
print("🛑 ERROR: X_train is not defined. Please run Block 5 first.")
else:
print("\n--- 6/10: Interaction Feature Engineering & Full XGBoost Training ---")
# ADVANCED: Create Interaction Feature
X_train['Loud_Energy_Interaction'] = X_train['num__loudness'] * X_train['num__energy']
X_test['Loud_Energy_Interaction'] = X_test['num__loudness'] * X_test['num__energy']
# Update feature names list
global feature_names
feature_names = X_train.columns
start_time = time.time()
# Core Model Parameters (from your PDF)
global xgb_model
xgb_model = xgb.XGBRegressor(
objective='reg:squarederror',
n_estimators=500,
learning_rate=0.05,
max_depth=7,
random_state=42,
n_jobs=-1
)
print("Starting training...")
xgb_model.fit(X_train, y_train)
global y_pred
y_pred = xgb_model.predict(X_test)
# Overfitting Check Metrics
global final_rmse
final_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
y_pred_train = xgb_model.predict(X_train)
train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
print(f"\n Training Complete. Test Set RMSE: {final_rmse:.2f}")
print(f" Training Set RMSE (Check): {train_rmse:.2f}")
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
if 'xgb_model' not in locals():
print("🛑 ERROR: xgb_model is not defined. Please run Block 6 first.")
else:
print("\n--- 7/10: Benchmark Model Comparison (PDF Step 7) ---")
# 1. Random Forest Regressor
rf_model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)
rf_rmse = np.sqrt(mean_squared_error(y_test, rf_pred))
# 2. Linear Regression (Baseline)
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
lr_pred = lr_model.predict(X_test)
lr_rmse = np.sqrt(mean_squared_error(y_test, lr_pred))
comparison_data = {
'Model': ['XGBoost Regressor (Optimized)', 'Linear Regression (Baseline)', 'Random Forest Regressor'],
'RMSE': [final_rmse, lr_rmse, rf_rmse]
}
comparison_df = pd.DataFrame(comparison_data).sort_values(by='RMSE')
print("\n Model Performance Comparison (RMSE):")
print(comparison_df.to_markdown(index=False, floatfmt=".2f"))
# Benchmarking Visualization (From PDF)
plt.figure(figsize=(9, 6))
sns.barplot(x='Model', y='RMSE', data=comparison_df, palette='Spectral')
plt.title('RMSE Comparison: XGBoost vs. Benchmarks', fontsize=16)
plt.ylabel('Root Mean Squared Error (RMSE)', fontsize=12)
plt.xticks(rotation=15)
plt.show()
from sklearn.model_selection import KFold, cross_val_score
if 'xgb_model' not in locals() or 'X_processed_df' not in locals():
print("🛑 ERROR: Dependencies missing. Please run Blocks 5 and 6 first.")
else:
print("\n--- 8/10: Cross-Validation (PDF Step 6) and Residual Analysis ---")
# --- Cross-Validation (Stability Check) ---
kfold = KFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(
xgb.XGBRegressor(n_estimators=500, learning_rate=0.05, max_depth=7, random_state=42),
X_processed_df, y,
scoring='neg_mean_squared_error',
cv=kfold,
n_jobs=-1
)
cv_rmse_scores = np.sqrt(-cv_scores)
print(f"Average Cross-Validation RMSE: {cv_rmse_scores.mean():.2f}")
# CV Visualization (From PDF)
plt.figure(figsize=(8, 5))
plt.bar(range(1, 6), cv_rmse_scores, color='skyblue')
plt.axhline(cv_rmse_scores.mean(), color='r', linestyle='--', label=f'Mean RMSE: {cv_rmse_scores.mean():.2f}')
plt.title('Model Stability: RMSE Across 5 Cross-Validation Folds', fontsize=14)
plt.xlabel("Fold Number", fontsize=12)
plt.ylabel('RMSE', fontsize=12)
plt.legend()
plt.grid(axis='y')
--- 8/10: Cross-Validation (PDF Step 6) and Residual Analysis ---
Average Cross-Validation RMSE: 17.99
plt.show()
plt.figure(figsize=(10, 6))
sns.scatterplot(x=y_pred, y=residuals, alpha=0.6)
plt.axhline(y=0, color='r', linestyle='--')
plt.title('Residuals vs. Predicted Values', fontsize=16)
plt.xlabel('Predicted Popularity Score', fontsize=12)
plt.ylabel('Residuals (Actual- Predicted)', fontsize=12)
plt.grid(True)
plt.show()
  import shap
if 'xgb_model' not in locals() or 'y_pred' not in locals():
print("🛑 ERROR: Dependencies missing. Please run Block 6 first.")
else:
print("\n--- 9/10: Final Interpretability Plots (SHAP & Actual vs. Predicted) ---")
# --- Actual vs. Predicted Plot (Key Visualization from PDF) ---
plt.figure(figsize=(8, 8))
sns.scatterplot(x=y_test, y=y_pred, alpha=0.6)
min_val = min(y_test.min(), y_pred.min())
max_val = max(y_test.max(), y_pred.max())
plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2)
plt.title('Actual vs. Predicted Popularity (Optimized XGBoost)', fontsize=16)
plt.xlabel('Actual Popularity Score (0-100)', fontsize=12)
plt.ylabel('Predicted Popularity Score (0-100)', fontsize=12)
plt.grid(True)
plt.show()
# --- SHAP Analysis (Explaining Feature Impact) ---
print("\n--- SHAP Summary Plot (Feature Impact) ---")
explainer = shap.TreeExplainer(xgb_model)
X_test_sample = X_test.sample(n=3000, random_state=42)
shap_values = explainer.shap_values(X_test_sample)
shap.summary_plot(shap_values, X_test_sample, feature_names=X_test_sample.columns, show=False)
plt.title('SHAP Summary Plot: Optimized Model Feature Impact', fontsize=16)
plt.tight_layout()
plt.show()
