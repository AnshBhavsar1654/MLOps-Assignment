from flask import Flask, request, jsonify
from flask_sqlalchemy import SQLAlchemy
from flask_cors import CORS
from datetime import datetime
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
import json
import os
import time
from waitress import serve

app = Flask(__name__)
CORS(app)


REQUEST_COUNT = Counter(
    'db_requests_total', 'Total number of HTTP requests', ['method', 'endpoint', 'http_status']
)

REQUEST_LATENCY = Histogram(
    'db_request_latency_seconds', 'Request latency in seconds', ['endpoint']
)

DB_ENTRIES = Gauge(
    'db_predictions_total', 'Total number of stored predictions'
)

# Middleware hooks for Prometheus
@app.before_request
def before_request():
    request.start_time = time.time()

@app.after_request
def after_request(response):
    latency = time.time() - getattr(request, "start_time", time.time())
    REQUEST_LATENCY.labels(request.path).observe(latency)
    REQUEST_COUNT.labels(request.method, request.path, response.status_code).inc()
    # Update prediction count gauge
    try:
        DB_ENTRIES.set(Prediction.query.count())
    except Exception:
        pass
    return response


# ============================================================
# 🔹 Database Configuration
# ============================================================

basedir = os.path.abspath(os.path.dirname(__file__))

env_database_url = os.environ.get('DATABASE_URL')
env_db_file = os.environ.get('DB_FILE')

if env_database_url:
    app.config['SQLALCHEMY_DATABASE_URI'] = env_database_url
else:
    db_file = env_db_file if env_db_file else os.path.join(basedir, 'predictions.db')
    os.makedirs(os.path.dirname(db_file), exist_ok=True)
    app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///' + db_file

app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['JSON_SORT_KEYS'] = False

db = SQLAlchemy(app)


# ============================================================
# 🔹 Database Model
# ============================================================

class Prediction(db.Model):
    __tablename__ = 'predictions'
    
    id = db.Column(db.Integer, primary_key=True)
    timestamp = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)
    model_name = db.Column(db.String(100), nullable=False)
    predicted_value = db.Column(db.Float, nullable=False)
    input_features = db.Column(db.Text, nullable=False)
    actual_value = db.Column(db.Float, nullable=True)
    
    def __repr__(self):
        return f'<Prediction {self.id}: {self.predicted_value} kWh>'
    
    def to_dict(self):
        return {
            'id': self.id,
            'timestamp': self.timestamp.isoformat(),
            'model_name': self.model_name,
            'predicted_value': self.predicted_value,
            'input_features': json.loads(self.input_features),
            'actual_value': self.actual_value
        }


# ============================================================
# 🔹 Initialize Database
# ============================================================

with app.app_context():
    db.create_all()
    print("✓ Database tables created successfully!")


# ============================================================
# 🔹 Routes
# ============================================================

@app.route('/', methods=['GET'])
def home():
    return jsonify({
        'service': 'Energy Consumption Database Service',
        'status': 'running',
        'database': app.config['SQLALCHEMY_DATABASE_URI'],
        'total_predictions': Prediction.query.count()
    })


@app.route('/metrics', methods=['GET'])
def metrics():
    """Expose Prometheus metrics"""
    return generate_latest(), 200, {'Content-Type': CONTENT_TYPE_LATEST}


@app.route('/predictions', methods=['POST'])
def add_prediction():
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        required_fields = ['model_name', 'predicted_value', 'input_features']
        for field in required_fields:
            if field not in data:
                return jsonify({'error': f'Missing required field: {field}'}), 400
        
        new_prediction = Prediction(
            model_name=data['model_name'],
            predicted_value=float(data['predicted_value']),
            input_features=json.dumps(data['input_features']),
            actual_value=float(data['actual_value']) if data.get('actual_value') else None
        )
        db.session.add(new_prediction)
        db.session.commit()
        
        DB_ENTRIES.set(Prediction.query.count())
        
        return jsonify({
            'message': 'Prediction stored successfully',
            'prediction_id': new_prediction.id,
            'status': 'success'
        }), 201
    
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': str(e), 'status': 'failed'}), 400


@app.route('/predictions', methods=['GET'])
def get_predictions():
    try:
        limit = request.args.get('limit', 50, type=int)
        offset = request.args.get('offset', 0, type=int)
        model_name = request.args.get('model_name', None)
        
        query = Prediction.query
        if model_name:
            query = query.filter_by(model_name=model_name)
        
        total_count = query.count()
        predictions = query.order_by(Prediction.timestamp.desc()).limit(limit).offset(offset).all()
        
        return jsonify({
            'predictions': [p.to_dict() for p in predictions],
            'total_count': total_count,
            'limit': limit,
            'offset': offset,
            'status': 'success'
        })
    except Exception as e:
        return jsonify({'error': str(e), 'status': 'failed'}), 400


@app.route('/predictions/<int:prediction_id>', methods=['GET'])
def get_prediction(prediction_id):
    try:
        prediction = Prediction.query.get(prediction_id)
        if not prediction:
            return jsonify({'error': 'Prediction not found'}), 404
        return jsonify({'prediction': prediction.to_dict(), 'status': 'success'})
    except Exception as e:
        return jsonify({'error': str(e), 'status': 'failed'}), 400


@app.route('/predictions/<int:prediction_id>', methods=['PUT'])
def update_prediction(prediction_id):
    try:
        prediction = Prediction.query.get(prediction_id)
        if not prediction:
            return jsonify({'error': 'Prediction not found'}), 404
        
        data = request.get_json()
        if 'actual_value' in data:
            prediction.actual_value = float(data['actual_value'])
        
        db.session.commit()
        return jsonify({
            'message': 'Prediction updated successfully',
            'prediction': prediction.to_dict(),
            'status': 'success'
        })
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': str(e), 'status': 'failed'}), 400


@app.route('/predictions/<int:prediction_id>', methods=['DELETE'])
def delete_prediction(prediction_id):
    try:
        prediction = Prediction.query.get(prediction_id)
        if not prediction:
            return jsonify({'error': 'Prediction not found'}), 404
        db.session.delete(prediction)
        db.session.commit()
        DB_ENTRIES.set(Prediction.query.count())
        return jsonify({'message': 'Prediction deleted successfully', 'status': 'success'})
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': str(e), 'status': 'failed'}), 400


@app.route('/predictions/stats', methods=['GET'])
def get_statistics():
    try:
        total_predictions = Prediction.query.count()
        avg_prediction = db.session.query(db.func.avg(Prediction.predicted_value)).scalar()
        min_prediction = db.session.query(db.func.min(Prediction.predicted_value)).scalar()
        max_prediction = db.session.query(db.func.max(Prediction.predicted_value)).scalar()
        model_counts = db.session.query(
            Prediction.model_name, db.func.count(Prediction.id)
        ).group_by(Prediction.model_name).all()
        
        return jsonify({
            'total_predictions': total_predictions,
            'average_prediction': round(avg_prediction, 2) if avg_prediction else 0,
            'min_prediction': round(min_prediction, 2) if min_prediction else 0,
            'max_prediction': round(max_prediction, 2) if max_prediction else 0,
            'predictions_by_model': {m: c for m, c in model_counts},
            'status': 'success'
        })
    except Exception as e:
        return jsonify({'error': str(e), 'status': 'failed'}), 400


@app.route('/predictions/clear', methods=['DELETE'])
def clear_predictions():
    try:
        count = Prediction.query.count()
        Prediction.query.delete()
        db.session.commit()
        DB_ENTRIES.set(0)
        return jsonify({'message': f'Cleared {count} predictions', 'status': 'success'})
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': str(e), 'status': 'failed'}), 400


# ============================================================
# 🔹 Start Server
# ============================================================

if __name__ == '__main__':
    print("\n" + "="*50)
    print("Energy Consumption Database Service with Prometheus Metrics")
    print("="*50)
    print(f"Database: {app.config['SQLALCHEMY_DATABASE_URI']}")
    print("Starting server on http://localhost:5002")
    print("="*50 + "\n")
    
    serve(app, host='0.0.0.0', port=5002)
