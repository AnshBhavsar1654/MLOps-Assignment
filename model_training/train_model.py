import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.pipeline import Pipeline
import mlflow
import mlflow.sklearn
import os

file_path = '/kaggle/input/energy-consumption-prediction/Energy_consumption.csv'
energy_data = pd.read_csv(file_path)


# FEATURE ENGINEERING

# Convert Timestamp to datetime and extract temporal features
energy_data['Timestamp'] = pd.to_datetime(energy_data['Timestamp'])
energy_data['Hour'] = energy_data['Timestamp'].dt.hour
energy_data['Day'] = energy_data['Timestamp'].dt.day
energy_data['Month'] = energy_data['Timestamp'].dt.month
energy_data['DayOfWeek'] = energy_data['Timestamp'].dt.dayofweek

# Create combined feature for special days
energy_data['daysComb'] = (
    (energy_data['DayOfWeek'].isin([3, 4, 5, 6])) | 
    (energy_data['Day'] >= 20) | 
    (energy_data['Day'] <= 10) | 
    (energy_data['Holiday'] == 'Yes')
).astype(int)

# Drop unnecessary columns (only if they exist)
columns_to_drop = ['Timestamp', 'Year', 'DayOfWeek', 'Day', 'Holiday']
columns_to_drop = [col for col in columns_to_drop if col in energy_data.columns]
energy_data = energy_data.drop(columns=columns_to_drop)


# ENCODING

# One-hot encode categorical variables
energy_data = pd.get_dummies(
    energy_data, 
    columns=['HVACUsage', 'LightingUsage'], 
    drop_first=True
)

# Convert boolean columns to integers
boolean_columns = energy_data.select_dtypes(include='bool').columns
energy_data[boolean_columns] = energy_data[boolean_columns].astype(int)

# TRAIN-TEST SPLIT
X = energy_data.drop(columns=['EnergyConsumption'])
y = energy_data['EnergyConsumption']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# MODEL TRAINING - RANDOM FOREST

# Define parameter grid
param_grid_rf = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Create a simple pipeline with the model
pipeline = Pipeline([
    ('model', RandomForestRegressor(random_state=42))
])

# Update param grid for pipeline (prefix with 'model__')
param_grid_rf = {
    'model__n_estimators': [100, 200, 300],
    'model__max_depth': [10, 20, None],
    'model__min_samples_split': [2, 5, 10],
    'model__min_samples_leaf': [1, 2, 4]
}

# Initialize GridSearchCV with pipeline
rf_grid = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid_rf,
    scoring='neg_mean_squared_error',
    cv=5,
    verbose=1,
    n_jobs=-1
)

# Set tracking URI from env if present (especially for Docker/Jenkins environments)
mlflow_uri = os.environ.get("MLFLOW_TRACKING_URI")
if mlflow_uri:
    mlflow.set_tracking_uri(mlflow_uri)

mlflow.set_experiment("Energy_Consumption_RF")

with mlflow.start_run():
    # Run the existing training and evaluation code inside this context
    # Fit the model
    rf_grid.fit(X_train, y_train)

    # Get best model
    best_model = rf_grid.best_estimator_

    # Evaluate
    y_pred = best_model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # Log parameters, score, and artifact to MLflow
    mlflow.log_params(rf_grid.best_params_)
    mlflow.log_metric('mae', mae)
    mlflow.log_metric('mse', mse)
    mlflow.log_metric('rmse', np.sqrt(mse))
    mlflow.log_metric('r2_score', r2)
    mlflow.sklearn.log_model(best_model, "random_forest_model")
    mlflow.log_artifact('energy_consumption_model.pkl')

# SAVE MODEL AND PREPROCESSING INFO
print("\nSaving model and preprocessing information...")

# Create a dictionary with all necessary components
model_package = {
    'model': best_model,
    'model_name': 'Random Forest',
    'feature_names': X.columns.tolist(),
    'scaler': None,  # We didn't use scaling for tree-based models
    'metrics': {
        'r2_score': r2,
        'mse': mse,
        'mae': mae
    }
}

# Save the complete package
with open('energy_consumption_model.pkl', 'wb') as f:
    pickle.dump(model_package, f)

print("Model saved as 'energy_consumption_model.pkl'")

# SAVE FEATURE NAMES SEPARATELY (OPTIONAL)
with open('feature_names.pkl', 'wb') as f:
    pickle.dump(X.columns.tolist(), f)

print("Feature names saved as 'feature_names.pkl'")

# LOAD AND USE MODEL
print("\n=== Loading Model Demo ===")

# Load the model
with open('energy_consumption_model.pkl', 'rb') as f:
    loaded_package = pickle.load(f)

loaded_model = loaded_package['model']
print(f"Loaded model: {loaded_package['model_name']}")
print(f"Model R² Score: {loaded_package['metrics']['r2_score']:.4f}")

# Make a sample prediction
sample_data = X_test.iloc[:5]
predictions = loaded_model.predict(sample_data)

print("\nSample Predictions:")
for i, pred in enumerate(predictions):
    print(f"Sample {i+1}: Predicted = {pred:.2f}, Actual = {y_test.iloc[i]:.2f}")

print("\n Model training and export complete")