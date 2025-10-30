# Energy Consumption Prediction - Microservices Architecture

This project implements an energy consumption prediction system using a microservices architecture with three independent services.

## 🏗️ Architecture Overview

```
┌─────────────────┐
│    Frontend     │  (Static HTML)
│   HTML + JS     │
└────────┬────────┘
         │
         ├──────────────────┐
         │                  │
┌────────▼────────┐  ┌──────▼──────────┐
│    Backend      │  │   DB Service    │
│  (Port 5001)    │──┤   (Port 5002)   │
│  ML Prediction  │  │   SQLite DB     │
└─────────────────┘  └─────────────────┘
```

## 📁 Project Structure

```
services/
├── frontend/
│   └── index.html          # Static UI interface
├── backend/
│   ├── app.py              # ML prediction service
│   └── requirements.txt
├── db_service/
│   ├── app.py              # Database service
│   ├── predictions.db      # SQLite database (auto-created)
│   └── requirements.txt
└── README.md
```

## 🚀 Quick Start

### Step 1: Install Dependencies

Open **two separate terminals** and navigate to each service directory:

#### Terminal 1 - Backend
```bash
cd services/backend
pip install -r requirements.txt
```

#### Terminal 2 - Database Service
```bash
cd services/db_service
pip install -r requirements.txt
```

### Step 2: Start Backend Services

**IMPORTANT:** Start services in this order:

#### 1. Start Database Service (Terminal 2)
```bash
cd services/db_service
python app.py
```
✓ Running on: `http://localhost:5002`

#### 2. Start Backend Service (Terminal 1)
```bash
cd services/backend
python app.py
```
✓ Running on: `http://localhost:5001`

### Step 3: Access the Application

Simply open the HTML file in your browser:
- **Option 1:** Double-click `services/frontend/index.html`
- **Option 2:** Right-click → Open with → Your browser
- **Option 3:** Use Live Server extension in VS Code

## 🎯 Service Details

### 1. Frontend (Static HTML)
- **Purpose:** User interface for making predictions
- **Technology:** Vanilla HTML, CSS, JavaScript
- **Features:**
  - Input form for energy consumption parameters
  - Real-time prediction results
  - Prediction history table
  - Statistics dashboard
  - Responsive design
  - No server required - runs directly in browser

### 2. Backend Service (Port 5001)
- **Purpose:** ML model inference and predictions
- **Features:**
  - Load pre-trained Random Forest model
  - Process input features
  - Make predictions
  - Send predictions to database service

**API Endpoints:**
- `GET /` - Health check
- `GET /model-info` - Model information and metrics
- `POST /predict` - Make single prediction
- `POST /batch-predict` - Make multiple predictions

### 3. Database Service (Port 5002)
- **Purpose:** Store and retrieve prediction history
- **Features:**
  - SQLite database
  - CRUD operations for predictions
  - Statistics and analytics
  - Prediction history

**API Endpoints:**
- `GET /` - Health check
- `POST /predictions` - Store new prediction
- `GET /predictions` - Get all predictions (with pagination)
- `GET /predictions/<id>` - Get specific prediction
- `PUT /predictions/<id>` - Update prediction
- `DELETE /predictions/<id>` - Delete prediction
- `GET /predictions/stats` - Get statistics
- `DELETE /predictions/clear` - Clear all predictions

## 📊 Making Predictions

### Using the Web Interface

1. Open `services/frontend/index.html` in your browser
2. Fill in the form with values:
   - **Temperature:** Ambient temperature (°C)
   - **Humidity:** Humidity percentage (%)
   - **Square Footage:** Building size (sq ft)
   - **Occupancy:** Number of occupants
   - **HVAC Usage:** On/Off
   - **Lighting Usage:** Low/Medium/High
   - **Renewable Energy:** Renewable energy contribution (kWh)
   - **Hour:** Hour of the day (0-23)
   - **Month:** Month of the year (1-12)
   - **Holiday:** Yes/No

3. Click "Predict Energy Consumption"
4. View the predicted energy consumption in kWh

### Using API (cURL)

```bash
curl -X POST http://localhost:5001/predict \
  -H "Content-Type: application/json" \
  -d '{
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
  }'
```

## 🧪 Testing Services

### Test Backend Service
```bash
curl http://localhost:5001/
curl http://localhost:5001/model-info
```

### Test Database Service
```bash
curl http://localhost:5002/
curl http://localhost:5002/predictions/stats
curl http://localhost:5002/predictions?limit=5
```

## 📝 Model Information

- **Model Type:** Random Forest Regressor
- **Target:** Energy Consumption (kWh)
- **Features:** 
  - Temporal features (Hour, Month)
  - Environmental features (Temperature, Humidity)
  - Building features (SquareFootage, Occupancy)
  - Usage patterns (HVAC, Lighting)
  - Renewable energy contribution

## 🔧 Troubleshooting

### Port Already in Use
If you get a port error for backend services:
1. Kill the process using that port
2. Change the port in the respective `app.py` file

**Windows:**
```powershell
netstat -ano | findstr :5001
taskkill /PID <PID> /F
```

### Model Not Loading
- Ensure `energy_consumption_model.pkl` exists in the root directory
- Check the path in `services/backend/app.py`

### Database Connection Issues
- Database file `predictions.db` will be auto-created
- Check write permissions in `services/db_service/` directory

### CORS Errors
- Ensure `flask-cors` is installed
- All services have CORS enabled by default

## 🔐 Security Notes

For production deployment:
- Add authentication and authorization
- Use environment variables for configuration
- Enable HTTPS
- Add rate limiting
- Validate and sanitize all inputs
- Use a production database (PostgreSQL, MySQL)
- Add logging and monitoring

## 📦 Dependencies

**Backend Services:**
- **Flask:** Web framework
- **Flask-CORS:** Cross-origin resource sharing
- **Flask-SQLAlchemy:** Database ORM (db_service only)
- **pandas, numpy, scikit-learn:** ML dependencies (backend only)

**Frontend:**
- No dependencies - pure HTML/CSS/JavaScript

## 🎨 Features

✅ Microservices architecture  
✅ RESTful API design  
✅ Modern, responsive UI  
✅ Real-time predictions  
✅ Prediction history  
✅ Statistics dashboard  
✅ Error handling  
✅ Loading states  
✅ SQLite database  
✅ CORS enabled  

## 📞 Support

If you encounter any issues:
1. Check all services are running
2. Verify model files exist
3. Check terminal logs for errors
4. Ensure all dependencies are installed

---

**Created for MLOps Assignment - Nirma University**
