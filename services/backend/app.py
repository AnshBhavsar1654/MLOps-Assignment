from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import pandas as pd
import numpy as np
import os
import requests
from waitress import serve
import mlflow
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
import time

app = Flask(__name__)
# Prometheus metrics
REQUEST_COUNT = Counter(
    'backend_requests_total', 'Total number of requests received', ['method', 'endpoint', 'http_status']
)

REQUEST_LATENCY = Histogram(
    'backend_request_latency_seconds', 'Request latency in seconds', ['endpoint']
)

PREDICTION_COUNT = Counter(
    'backend_predictions_total', 'Total number of predictions made', ['endpoint']
)

CORS(app)

@app.before_request
def before_request():
    request.start_time = time.time()

@app.after_request
def after_request(response):
    latency = time.time() - getattr(request, "start_time", time.time())
    REQUEST_LATENCY.labels(request.path).observe(latency)
    REQUEST_COUNT.labels(request.method, request.path, response.status_code).inc()
    return response


# Load the trained model
DEFAULT_MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', '..', 'energy_consumption_model.pkl')
DEFAULT_FEATURE_NAMES_PATH = os.path.join(os.path.dirname(__file__), '..', '..', 'feature_names.pkl')

# Allow overriding via environment (useful for Docker)
MODEL_PATH = os.environ.get("MODEL_PATH", DEFAULT_MODEL_PATH)
FEATURE_NAMES_PATH = os.environ.get("FEATURE_NAMES_PATH", DEFAULT_FEATURE_NAMES_PATH)

# Database service URL (overridable for Docker networking)
DB_SERVICE_URL = os.environ.get("DB_SERVICE_URL", "http://localhost:5002/predictions")

# Set MLflow tracking URI for Docker/Jenkins
mlflow_uri = os.environ.get("MLFLOW_TRACKING_URI")
if mlflow_uri:
    mlflow.set_tracking_uri(mlflow_uri)
mlflow.set_experiment("Energy_Consumption_Inference_API")

# Load model at startup
try:
    with open(MODEL_PATH, 'rb') as f:
        model_package = pickle.load(f)
    
    model = model_package['model']
    feature_names = model_package['feature_names']
    model_metrics = model_package['metrics']
    
    print("✓ Model loaded successfully!")
    print(f"Model Type: {model_package['model_name']}")
    print(f"R² Score: {model_metrics['r2_score']:.4f}")
    print(f"Features: {len(feature_names)}")
except Exception as e:
    print(f"Error loading model: {str(e)}")
    model = None
    feature_names = None


@app.route('/', methods=['GET'])
def home():
    """Health check endpoint"""
    return jsonify({
        'service': 'Energy Consumption Prediction Backend',
        'status': 'running',
        'model_loaded': model is not None
    })


@app.route('/model-info', methods=['GET'])
def model_info():
    """Get model information and metrics"""
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
    return jsonify({
        'model_name': model_package['model_name'],
        'feature_names': feature_names,
        'metrics': model_metrics,
        'total_features': len(feature_names)
    })
@app.route('/metrics', methods=['GET'])
def metrics():
    from flask import Response
    return Response(generate_latest(), mimetype=CONTENT_TYPE_LATEST)

@app.route('/predict', methods=['POST'])
def predict():
    """
    Predict energy consumption based on input features
    
    Expected JSON format:
    {
        "Temperature": 25.5,
        "Humidity": 60,
        "SquareFootage": 2000,
        "Occupancy": 4,
        "HVACUsage": "On",
        "LightingUsage": "High",
        "RenewableEnergy": 10,
        "Hour": 14,
        "Month": 6,
        "Holiday": "No"
    }
    """
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
    try:
        # Get input data
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No input data provided'}), 400
        
        # Process input data
        input_df = preprocess_input(data)
        
        # Make prediction
        prediction = model.predict(input_df)[0]
        prediction = float(prediction)
        
        # MLflow Logging (optional for monitoring)
        try:
            mlflow.log_param("endpoint", "predict")
            mlflow.log_dict(data, "input_features.json")
            mlflow.log_metric("prediction", prediction)
        except Exception as mlflow_err:
            print(f"MLflow logging error: {mlflow_err}")
        
        # Prepare response
        response = {
            'prediction': round(prediction, 2),
            'unit': 'kWh',
            'input_data': data,
            'status': 'success'
        }
        
        # Store prediction in database service (async - don't wait for response)
        try:
            prediction_data = {
                'input_features': data,
                'predicted_value': round(prediction, 2),
                'model_name': model_package['model_name']
            }
            requests.post(DB_SERVICE_URL, json=prediction_data, timeout=2)
        except Exception as db_error:
            print(f"Warning: Could not store prediction in database: {str(db_error)}")
        PREDICTION_COUNT.labels(endpoint='/predict').inc()

        return jsonify(response)
    
    except Exception as e:
        return jsonify({
            'error': str(e),
            'status': 'failed'
        }), 400


def preprocess_input(data):
    """
    Preprocess input data to match model's expected format
    """
    # Create initial dataframe
    df = pd.DataFrame([data])
    
    # Convert Timestamp if present, or use provided temporal features
    if 'Timestamp' in df.columns:
        df['Timestamp'] = pd.to_datetime(df['Timestamp'])
        df['Hour'] = df['Timestamp'].dt.hour
        df['Day'] = df['Timestamp'].dt.day
        df['Month'] = df['Timestamp'].dt.month
        df['DayOfWeek'] = df['Timestamp'].dt.dayofweek
        df = df.drop(columns=['Timestamp'])
    
    # Create daysComb feature if we have the required fields
    if all(col in df.columns for col in ['DayOfWeek', 'Day', 'Holiday']):
        df['daysComb'] = (
            (df['DayOfWeek'].isin([3, 4, 5, 6])) | 
            (df['Day'] >= 20) | 
            (df['Day'] <= 10) | 
            (df['Holiday'] == 'Yes')
        ).astype(int)
        
        # Drop columns that were used to create daysComb
        df = df.drop(columns=['DayOfWeek', 'Day', 'Holiday'], errors='ignore')
    elif 'Holiday' in df.columns:
        # If daysComb should be created but we don't have all fields
        # Just drop Holiday if present
        df = df.drop(columns=['Holiday'], errors='ignore')
    
    # One-hot encode categorical variables
    if 'HVACUsage' in df.columns:
        df = pd.get_dummies(df, columns=['HVACUsage'], drop_first=True)
    
    if 'LightingUsage' in df.columns:
        df = pd.get_dummies(df, columns=['LightingUsage'], drop_first=True)
    
    # Ensure all expected features are present
    for feature in feature_names:
        if feature not in df.columns:
            df[feature] = 0
    
    # Reorder columns to match training data
    df = df[feature_names]
    
    return df


@app.route('/batch-predict', methods=['POST'])
def batch_predict():
    """
    Make predictions for multiple samples
    
    Expected JSON format:
    {
        "samples": [
            {...sample1...},
            {...sample2...}
        ]
    }
    """
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
    try:
        data = request.get_json()
        samples = data.get('samples', [])
        
        if not samples:
            return jsonify({'error': 'No samples provided'}), 400
        
        predictions = []
        for sample in samples:
            input_df = preprocess_input(sample)
            prediction = model.predict(input_df)[0]
            predictions.append({
                'input': sample,
                'prediction': round(float(prediction), 2)
            })
            try:
                mlflow.log_param("endpoint", "batch-predict")
                mlflow.log_dict(sample, "input_features_batch.json")
                mlflow.log_metric("prediction", float(prediction))
            except Exception as mlflow_err:
                print(f"MLflow logging error: {mlflow_err}")
        PREDICTION_COUNT.labels(endpoint='/batch-predict').inc()

        return jsonify({
            'predictions': predictions,
            'count': len(predictions),
            'status': 'success'
        })
    
    except Exception as e:
        return jsonify({
            'error': str(e),
            'status': 'failed'
        }), 400


if __name__ == '__main__':
    print("\n" + "="*50)
    print("Energy Consumption Prediction Backend Service")
    print("="*50)
    print(f"Model Status: {'Loaded ✓' if model else 'Not Loaded ✗'}")
    print("Starting server on http://localhost:5001")
    print("="*50 + "\n")
    
    serve(app, host='0.0.0.0', port=5001)