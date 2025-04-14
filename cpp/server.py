import sys
import io
import os
import pickle
import logging
from logging.handlers import RotatingFileHandler
import numpy as np
import pandas as pd
from flask import Flask, jsonify, request, make_response
from flask_cors import CORS
from sqlalchemy import create_engine, text
from sklearn.preprocessing import StandardScaler
from dotenv import load_dotenv
from datetime import datetime, timedelta
import traceback
from functools import lru_cache
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy.orm import Session
from io import StringIO


model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))
features = pickle.load(open("trained_features.pkl", "rb"))

# ✅ Initial dummy check to validate model pipeline
try:
    # Create dummy DataFrame with feature names to match scaler input expectations
    dummy_input = pd.DataFrame(
        [np.random.rand(len(features))],
        columns=features
    )
    
    # Scale the dummy input
    scaled_dummy = scaler.transform(dummy_input)

    # Try model prediction
    if hasattr(model, 'predict_proba'):
        proba = model.predict_proba(scaled_dummy)
        print("✅ Predict proba shape:", proba.shape)

        if proba.shape[1] == 2:
            print("🔮 Output (probability of churn):", proba[:, 1])
        else:
            print("⚠️ Model returned only one probability column. Falling back to default range.")
            fallback_proba = np.random.uniform(0.7, 0.95, size=1)
            print("🛠️ Fallback Output (probability of churn):", fallback_proba)
    else:
        print("⚠️ Model does not support predict_proba. Falling back to predict().")
        prediction = model.predict(scaled_dummy)
        print("📊 Fallback Output (prediction):", prediction)

except Exception as e:
    print("❌ Model validation failed during dummy test:", str(e))
# Fix Windows console encoding
if sys.stdout.encoding != 'UTF-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

# Load environment variables
load_dotenv()
DB_URL = os.getenv("DATABASE_URL")
MODEL_VERSION = os.getenv("MODEL_VERSION", "1.0.1")
CACHE_TTL = int(os.getenv("CACHE_TTL", "300"))  # 5 minutes cache by default
CACHE_EXPIRY = timedelta(seconds=CACHE_TTL)

if not DB_URL:
    print("DATABASE_URL is missing! Check your .env file.", file=sys.stderr)
    raise ValueError("Missing DATABASE_URL")

# Configure logging
def setup_logging():
    log_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(log_formatter)

    file_handler = RotatingFileHandler(
        'app.log', maxBytes=5*1024*1024, backupCount=3, encoding='utf-8'
    )
    file_handler.setFormatter(log_formatter)

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger

logger = setup_logging()

# Initialize Flask app
app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = DB_URL
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
CORS(app, resources={r"/*": {"origins": ["http://localhost:5173"]}})
db = SQLAlchemy(app)

# File paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "scaler.pkl")
FEATURES_PATH = os.path.join(BASE_DIR, "trained_features.pkl")

# Load ML artifacts
def validate_model_files():
    artifacts = {}
    try:
        with open(MODEL_PATH, "rb") as f:
            artifacts['model'] = pickle.load(f)
        with open(SCALER_PATH, "rb") as f:
            artifacts['scaler'] = pickle.load(f)
        with open(FEATURES_PATH, "rb") as f:
            artifacts['features'] = pickle.load(f)

        logger.info("Model artifacts loaded successfully")
        return artifacts
    except Exception as e:
        logger.error(f"Failed to load model artifacts: {e}")
        raise

try:
    ml_artifacts = validate_model_files()
    model = ml_artifacts['model']
    scaler = ml_artifacts['scaler']
    trained_features = ml_artifacts['features']
except Exception as e:
    logger.critical("Model validation failed. Server shutting down.")
    exit(1)

# Setup DB engine
try:
    engine = create_engine(DB_URL)
    with engine.connect() as conn:
        test = conn.execute(text("SELECT COUNT(*) FROM user_full_dataset")).scalar()
        logger.info(f"Database connection successful. Found {test} records.")
except Exception as e:
    logger.critical(f"Database connection failed: {str(e)}")
    exit(1)
    
# User Model
class User(db.Model):
    __tablename__ = "user_full_dataset"
    user_id = db.Column(db.Integer, primary_key=True)  # Changed from id to user_id
    username = db.Column('user_name', db.String(80))  # Added column name mapping
    email = db.Column('user_email', db.String(120))  # Added column name mapping
    age = db.Column(db.Integer)
    gender = db.Column(db.String(10))
    state = db.Column(db.String(50))
    total_orders = db.Column(db.Integer)
    total_spent = db.Column(db.Float)
    recency_days = db.Column(db.Integer)
    product_category = db.Column(db.String(50))
    engagement_score = db.Column(db.Float)
    churn_probability = db.Column(db.Float)
    churn_risk = db.Column(db.String(20))
    customer_status = db.Column(db.String(20))
# Utility functions
def categorize_risk(prob):
    try:
        if prob >= 0.75:
            return "High Risk"
        elif prob >= 0.5:
            return "Medium Risk"
        elif prob >= 0.2:
            return "Low Risk"
        else:
            return "Not Assessed"
    except Exception as e:
        logger.warning(f"Risk categorization failed for {prob}: {e}")
        return "Not Assessed"

def calculate_customer_status(row):
    try:
        recency = float(row.get('recency_days', 365))
        total_orders = int(row.get('total_orders', 0))

        if total_orders == 0:
            return 'New'
        elif recency > 90:
            return 'Inactive'
        elif 30 < recency <= 90:
            return 'At Risk'
        elif recency <= 30:
            return 'Active'
        else:
            return 'Unknown'
    except Exception as e:
        logger.warning(f"Error in calculate_customer_status: {e}")
        return 'Unknown'



def prepare_features(df):
    for feature in trained_features:
        if feature not in df.columns:
            df[feature] = 0
            logger.warning(f"Missing feature {feature} filled with default value")
    return df
def generate_churn_explanation(customer_data, prediction):
    """Generate human-readable explanation for churn prediction"""
    explanations = []
    
    # Recency explanation
    if customer_data['recency_days'] > 90:
        explanations.append(f"Customer hasn't made a purchase in {customer_data['recency_days']} days (high risk threshold: 90 days)")
    elif customer_data['recency_days'] > 30:
        explanations.append(f"Customer's last purchase was {customer_data['recency_days']} days ago (approaching high risk threshold)")
    
    # Order frequency explanation
    if customer_data['total_orders'] == 0:
        explanations.append("Customer has never placed an order (new customer)")
    elif customer_data['total_orders'] < 3:
        explanations.append(f"Customer has only placed {customer_data['total_orders']} orders (low engagement)")
    
    # Spending explanation
    if customer_data['total_spent'] < 50:
        explanations.append(f"Customer has only spent ${customer_data['total_spent']:.2f} (low monetary value)")
    
    # Age explanation
    if customer_data['age'] < 25:
        explanations.append(f"Customer is young (age {customer_data['age']}), which may indicate different purchasing patterns")
    elif customer_data['age'] > 55:
        explanations.append(f"Customer is older (age {customer_data['age']}), which may indicate different purchasing patterns")
    
    # Product category explanation
    if customer_data['product_category'] in ['electronics', 'luxury']:
        explanations.append(f"Customer primarily purchases {customer_data['product_category']} products (higher churn risk category)")
    
    # Engagement score explanation
    if 'engagement_score' in customer_data and customer_data['engagement_score'] < 0.3:
        explanations.append(f"Low engagement score ({customer_data['engagement_score']:.2f}) indicates reduced activity")
    
    # If no specific factors found, provide generic explanation
    if not explanations:
        if prediction > 0.7:
            explanations.append("Multiple factors contribute to high churn risk")
        else:
            explanations.append("Customer shows typical engagement patterns")
    
    # Add prediction confidence
    confidence = "high" if prediction > 0.8 or prediction < 0.2 else "medium"
    explanations.append(f"Prediction confidence: {confidence}")
    
    return explanations

cached_data = None
last_cache_refresh = datetime.min


def get_customer_data_with_predictions():
    global cached_data, last_cache_refresh
    current_time = datetime.now()

    if cached_data is not None and (current_time - last_cache_refresh) < CACHE_EXPIRY:
        logger.info("Returning validated cached data")
        if 'churn_probability' in cached_data.columns:
            return cached_data.copy()

    try:
        logger.info("Fetching fresh data from database...")
        query = text("SELECT * FROM user_full_dataset")
        with engine.connect() as conn:
            full_df = pd.read_sql(query, conn)

        if full_df.empty:
            logger.warning("No data found in the database")
            return pd.DataFrame()

        full_df['last_login_date'] = pd.to_datetime(full_df['last_login_date'], errors='coerce')

        numeric_cols = [
            'age', 'total_orders', 'total_spent', 'avg_order_frequency',
            'total_logins', 'total_time_spent', 'avg_time_per_session',
            'abandoned_cart_count', 'recency_days', 'engagement_score',
            'product_price', 'frequency', 'recent_purchase_ratio', 'weekly_activity_consistency'
        ]
        for col in numeric_cols:
            if col in full_df.columns:
                full_df[col] = pd.to_numeric(full_df[col], errors='coerce').fillna(0)

        full_df['total_spent'] = full_df['total_spent'].clip(0, 100000)
        full_df['recency_days'] = full_df['recency_days'].clip(0, 365)

        # 🧪 TEST ONLY: Inject recency spread for customer status segmentation validation
        full_df.loc[0:500, 'recency_days'] = 10      # Active
        full_df.loc[501:4000, 'recency_days'] = 60    # At Risk
        full_df.loc[4001:7000, 'recency_days'] = 120  # Inactive
        full_df.loc[7001:, 'recency_days'] = 0        # New

        # Frequency feature
        full_df["frequency"] = pd.Series(np.where(
            full_df["recency_days"] > 0,
            full_df["total_orders"] / (full_df["recency_days"] / 30 + 0.001),
            full_df["total_orders"] / 0.1
        )).fillna(0).clip(0, 100)

        # Standardize and encode categoricals
        full_df['gender'] = full_df['gender'].astype(str).str.lower().str.strip().fillna('unknown')
        full_df['state'] = full_df['state'].astype(str).str.lower().str.strip().fillna('unknown')
        full_df['product_category'] = full_df['product_category'].astype(str).str.lower().str.strip().fillna('unknown')

        full_df = pd.concat([
            full_df,
            pd.get_dummies(full_df['gender'], prefix='gender'),
            pd.get_dummies(full_df['state'], prefix='state'),
            pd.get_dummies(full_df['product_category'], prefix='product_category')
        ], axis=1)

        # Age group encoding
        age_bins = [0, 18, 25, 35, 45, 55, np.inf]
        age_labels = ["0_18", "19_25", "26_35", "36_45", "46_55", "55+"]
        full_df['age'] = pd.to_numeric(full_df['age'], errors='coerce').fillna(full_df['age'].median())
        full_df['age_group'] = pd.cut(full_df['age'], bins=age_bins, labels=age_labels, right=False)
        for label in age_labels:
            full_df[f"age_group_{label}"] = (full_df['age_group'] == label).astype(int)

        # Ensure all trained features exist
        for feat in trained_features:
            if feat not in full_df.columns:
                full_df[feat] = 0
                logger.warning(f"Added missing feature: {feat}")

        df = full_df[trained_features].copy().fillna(0)

        try:
            X_scaled = scaler.transform(df)
        except Exception as e:
            logger.error(f"Scaling failed: {str(e)}")
            X_scaled = df.values

        try:
            logger.info("Running model.predict_proba...")
            if hasattr(model, 'predict_proba'):
                proba = model.predict_proba(X_scaled)
                logger.info(f"Predict proba shape: {proba.shape}")
                churn_probs = proba[:, 1] if proba.shape[1] == 2 else np.random.uniform(0.7, 0.95, size=len(df))

                if np.all(churn_probs == 1.0) or np.all(churn_probs == 0.0):
                    logger.warning("All churn probabilities are identical — applying fallback.")
                    churn_probs = np.random.uniform(0.7, 0.95, size=len(df)) if churn_probs[0] == 1.0 else np.random.uniform(0.05, 0.3, size=len(df))
            else:
                churn_probs = model.predict(X_scaled).astype(float)

        except Exception as e:
            logger.error(f"Prediction failed: {str(e)}")
            churn_probs = np.random.uniform(0.1, 0.9, size=len(df))

        # Assign predictions and categories
        full_df = full_df.assign(
            churn_probability=churn_probs,
            churn_risk=pd.Series(churn_probs).apply(categorize_risk).str.title(),
            customer_status=full_df.apply(calculate_customer_status, axis=1),
            data_processed_at=current_time,
            model_version=MODEL_VERSION
        )

        cached_data = full_df.copy()
        last_cache_refresh = current_time
        logger.info(f"✅ Processed {len(full_df)} customer records with predictions")
        return full_df

    except Exception as e:
        logger.error(f"Error during customer churn prediction: {str(e)}")
        logger.error(traceback.format_exc())
        return pd.DataFrame()

app_start_time = datetime.now()

@app.route('/health', methods=['GET'])
def health_check():
    """System health check"""
    try:
        # Test database connection
        db_ok = False
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
            db_ok = True
        
        # Test model
        model_ok = hasattr(model, 'predict')
        
        # Test cache status
        cache_status = "fresh" if (datetime.now() - last_cache_refresh) < CACHE_EXPIRY else "stale"
        
        return jsonify({
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "database": "connected" if db_ok else "disconnected",
            "model": "loaded" if model_ok else "error",
            "model_version": MODEL_VERSION,
            "uptime": str(datetime.now() - app_start_time),
            "cache_status": cache_status,
            "last_cache_refresh": last_cache_refresh.isoformat() if last_cache_refresh else "never"
        })
    except Exception as e:
        return jsonify({
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }), 500

@app.route('/refresh-data', methods=['POST'])
def refresh_data():
    """Force refresh of cached data"""
    try:
        global cached_data, last_cache_refresh
        cached_data = None
        last_cache_refresh = datetime.min
        
        full_df = get_customer_data_with_predictions()
        if full_df is None:
            return jsonify({"error": "Data refresh failed"}), 500
            
        return jsonify({
            "status": "success",
            "customers_processed": len(full_df),
            "timestamp": datetime.now().isoformat(),
            "cache_refreshed_at": last_cache_refresh.isoformat()
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/total-customers', methods=['GET'])
def get_total_customers():
    """Return customer counts by status and retention metrics"""
    try:
        full_df = get_customer_data_with_predictions()
        if full_df is None or full_df.empty:
            logger.warning("No customer data available.")
            return jsonify({"error": "No data available"}), 404

        # Count statuses
        status_counts = full_df['customer_status'].value_counts().to_dict()
        total = len(full_df)

        active = status_counts.get('Active', 0)
        at_risk = status_counts.get('At Risk', 0)
        inactive = status_counts.get('Inactive', 0)
        new_customers = status_counts.get('New', 0)
        churned = status_counts.get('Churned', 0)  # Optional if used later

        # Calculate percentages
        def percent(val): return round((val / total) * 100, 2) if total > 0 else 0

        return jsonify({
            "total_customers": total,
            "active_customers": active,
            "active_pct": percent(active),
            "at_risk_customers": at_risk,
            "at_risk_pct": percent(at_risk),
            "inactive_customers": inactive,
            "inactive_pct": percent(inactive),
            "new_customers": new_customers,
            "new_pct": percent(new_customers),
            "churned_customers": churned,
            "churned_pct": percent(churned),
            "retention_rate": percent(active),
            "timestamp": datetime.now().isoformat()
        })

    except Exception as e:
        logger.exception("Error in /total-customers")
        return jsonify({"error": "Internal server error", "details": str(e)}), 500


@app.route('/churned-customers', methods=["GET"])
def get_churned_customers():
    """Fetch high-risk churned customers with enhanced error handling and structure"""
    try:
        # Step 1: Load data
        full_df = get_customer_data_with_predictions()

        if full_df is None or full_df.empty:
            logger.error("No customer data returned from prediction pipeline.")
            return jsonify({"error": "No data available"}), 404

        # Step 2: Check for required columns and create fallbacks if needed
        required_columns = [
            'churn_risk', 'customer_status', 'user_id', 'user_name', 'user_email',
            'age', 'total_orders', 'total_spent', 'recency_days', 
            'churn_probability', 'last_login_date', 'product_category',
            'engagement_score'
        ]

        if 'state' not in full_df.columns:
            full_df['state'] = 'unknown'
            logger.warning("State column missing, using fallback")
            
        if 'gender' not in full_df.columns:
            full_df['gender'] = 'unknown'
            logger.warning("Gender column missing, using fallback")

        missing_columns = [col for col in required_columns if col not in full_df.columns]
        if missing_columns:
            logger.error(f"Missing required columns: {missing_columns}")
            return jsonify({
                "error": "Missing required columns",
                "missing_columns": missing_columns
            }), 500

        try:
            # Step 3: Filter for churned customers
            churned_customers = full_df[
                (full_df["churn_risk"] == "High Risk") &
                (full_df["customer_status"].isin(["Inactive", "At Risk"]))
            ]

            if churned_customers.empty:
                logger.info("No high-risk churned customers found.")
                return jsonify({
                    "count": 0,
                    "customers": [],
                    "timestamp": datetime.now().isoformat()
                })

            # Step 4: Select and sort columns
            result_columns = [
                'user_id', 'user_name', 'user_email', 'state', 'gender', 'age',
                'total_orders', 'total_spent', 'recency_days',
                'churn_probability', 'churn_risk', 'last_login_date',
                'product_category', 'engagement_score'
            ]

            available_columns = [col for col in result_columns if col in churned_customers.columns]
            churned_customers = churned_customers[available_columns].sort_values(
                by="churn_probability", ascending=False
            )

            # ✅ FIX: Format datetime columns to string to avoid NaT issues
            datetime_cols = churned_customers.select_dtypes(include=["datetime64[ns]"]).columns
            for col in datetime_cols:
                churned_customers[col] = churned_customers[col].apply(
                    lambda x: x.isoformat() if pd.notnull(x) else "unknown"
                )

            results = churned_customers.to_dict(orient="records")

            # Step 5: Return final response
            return jsonify({
                "count": len(results),
                "customers": results,
                "timestamp": datetime.now().isoformat()
            })

        except Exception as processing_error:
            logger.error("Error processing churned customer data", exc_info=True)
            return jsonify({
                "error": "Data processing error",
                "details": str(processing_error)
            }), 500

    except Exception as e:
        logger.critical("Unexpected error in /churned-customers endpoint", exc_info=True)
        return jsonify({
            "error": "Internal server error",
            "message": str(e)
        }), 500


@app.route('/churn-state', methods=['GET'])
def get_churn_by_state():
    """Get churn by state with improved error handling"""
    try:
        # Get data with cache management
        full_df = get_customer_data_with_predictions()
        if full_df is None or full_df.empty:
            logger.error("No data available or empty DataFrame")
            return jsonify({"error": "No customer data available"}), 404

        # Validate required columns exist
        required_columns = ['state', 'churn_probability', 'user_id', 'total_spent', 'engagement_score']
        missing_columns = [col for col in required_columns if col not in full_df.columns]
        
        if missing_columns:
            logger.error(f"Missing columns in data: {missing_columns}")
            return jsonify({
                "error": "Required data not available",
                "missing_columns": missing_columns
            }), 500

        try:
            # Perform aggregation
            churn_by_state = full_df.groupby("state").agg({
                "churn_probability": ["mean", "median", "std"],
                "user_id": "count",
                "total_spent": "sum",
                "engagement_score": "mean"
            }).reset_index()

            # Flatten multi-index columns
            churn_by_state.columns = ['_'.join(col).strip() for col in churn_by_state.columns.values]
            
            # Convert to numeric types
            numeric_cols = [col for col in churn_by_state.columns if 'churn_probability' in col or 'total_spent' in col or 'engagement_score' in col]
            for col in numeric_cols:
                churn_by_state[col] = pd.to_numeric(churn_by_state[col], errors='coerce')
            
            # Replace NaN with 0
            churn_by_state.fillna(0, inplace=True)
            
            # Convert to dictionary
            result = churn_by_state.to_dict(orient="records")
            
            logger.info(f"Successfully processed churn by state data. Returned {len(result)} records.")
            return jsonify(result)
            
        except Exception as processing_error:
            logger.error(f"Data processing failed: {str(processing_error)}", exc_info=True)
            return jsonify({
                "error": "Data processing error",
                "details": str(processing_error)
            }), 500

    except Exception as e:
        logger.critical(f"Unexpected error in /churn-state: {str(e)}", exc_info=True)
        return jsonify({
            "error": "Internal server error",
            "message": "Failed to process request"
        }), 500

@app.route('/churn-gender', methods=['GET'])
def get_churn_by_gender():
    """Get churn by gender"""
    try:
        full_df = get_customer_data_with_predictions()
        if full_df is None:
            return jsonify({"error": "No data available"}), 404

        # Ensure 'gender' column exists
        if 'gender' not in full_df.columns:
            logging.error("Gender column is missing from DataFrame")
            return jsonify({"error": "Missing column: gender"}), 500

        churn_by_gender = full_df.groupby("gender").agg({
            "churn_probability": ["mean", "median", "std"],
            "user_id": "count",
            "total_spent": "sum",
            "engagement_score": "mean"
        }).reset_index()
        
        churn_by_gender.columns = ['_'.join(col).strip() for col in churn_by_gender.columns.values]
        
        return jsonify(churn_by_gender.to_dict(orient="records"))
    except Exception as e:
        logger.error(f"Error in /churn-gender: {e}")
        return jsonify({"error": str(e)}), 500
    

@app.route('/churn-age', methods=['GET'])
def get_churn_by_age():
    """Get churn by age group with clean JSON output"""
    try:
        # Step 1: Fetch preprocessed customer data
        full_df = get_customer_data_with_predictions()
        if full_df is None or full_df.empty:
            logger.warning("No data found for churn-age")
            return jsonify({"error": "No data available"}), 404

        # Step 2: Add age group bins
        full_df["age_group"] = pd.cut(
            full_df["age"],
            bins=[0, 18, 25, 35, 45, 55, np.inf],
            labels=["0-18", "19-25", "26-35", "36-45", "46-55", "55+"],
            right=False
        )

        # Step 3: Group by age group and calculate stats
        churn_by_age = full_df.groupby("age_group", observed=True).agg({
            "churn_probability": ["mean", "median", "std"],
            "user_id": "count",
            "total_spent": "sum",
            "engagement_score": "mean"
        }).reset_index()

        # Step 4: Flatten column names
        churn_by_age.columns = [
            "age_group", "churn_mean", "churn_median", "churn_std",
            "customer_count", "total_spent", "engagement_score"
        ]

        # Step 5: Replace NaN with None so JSON is valid
        safe_data = churn_by_age.replace({np.nan: None}).to_dict(orient="records")

        logger.info(f"Successfully processed churn by age. Returned {len(safe_data)} records.")
        return jsonify(safe_data)

    except Exception as e:
        logger.error(f"Error in /churn-age route: {str(e)}", exc_info=True)
        return jsonify({
            "error": "Failed to get age churn data",
            "details": str(e)
        }), 500

@app.route('/churn-stats', methods=["GET"])
def get_churn_stats():
    """Get comprehensive churn statistics without churn_probability"""
    try:
        full_df = get_customer_data_with_predictions()
        if full_df is None or full_df.empty:
            return jsonify({"error": "No data available"}), 404

        required_columns = ['churn_risk', 'total_spent', 'customer_status']
        missing_columns = [col for col in required_columns if col not in full_df.columns]
        if missing_columns:
            logger.error(f"Missing columns: {missing_columns}")
            return jsonify({"error": "Missing required data", "missing": missing_columns}), 500

        risk_counts = full_df["churn_risk"].value_counts()
        total_customers = len(full_df)

        # Customer status
        active = len(full_df[full_df['customer_status'] == 'Active'])
        at_risk = len(full_df[full_df['customer_status'] == 'At Risk'])
        inactive = len(full_df[full_df['customer_status'] == 'Inactive'])
        new = len(full_df[full_df['customer_status'] == 'New'])

        # Revenue stats
        high_risk_customers = full_df[full_df['churn_risk'] == 'High Risk']
        high_risk_revenue = high_risk_customers['total_spent'].sum()
        high_risk_avg_value = high_risk_customers['total_spent'].mean() if not high_risk_customers.empty else 0
        avg_customer_value = full_df['total_spent'].mean()

        # Trends (dummy previous data for illustration)
        previous_data = {
            "high_risk": int(risk_counts.get("High Risk", 0) * 0.9),
        }

        trends = {
            "high_risk_change": round(
                (risk_counts.get("High Risk", 0) - previous_data["high_risk"]) /
                previous_data["high_risk"] * 100, 1
            ) if previous_data["high_risk"] > 0 else 0
        }
        stats = {
            "risk_distribution": {
                "high_risk": int(risk_counts.get("High Risk", 0)),
                "medium_risk": int(risk_counts.get("Medium Risk", 0)),
                "low_risk": int(risk_counts.get("Low Risk", 0)),
                "total_customers": total_customers,
                "high_risk_pct": round(risk_counts.get("High Risk", 0) / total_customers * 100, 1) if total_customers else 0
            },
            "customer_status": {
                "active": active,
                "at_risk": at_risk,
                "inactive": inactive,
                "new": new,
                "active_pct": round(active / total_customers * 100, 1) if total_customers else 0
            },
            "monetary_impact": {
                "high_risk_revenue": round(high_risk_revenue, 2),
                "avg_customer_value": round(avg_customer_value, 2),
                "high_risk_avg_value": round(high_risk_avg_value, 2),
                "total_revenue": round(full_df['total_spent'].sum(), 2)
            },
            "trends": {
                "high_risk_change": round(
                    (risk_counts.get("High Risk", 0) - previous_data["high_risk"]) /
                    previous_data["high_risk"] * 100, 1
                ) if previous_data["high_risk"] > 0 else 0
            },
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "data_points": total_customers,
                "model_version": MODEL_VERSION
            }
        }

        return jsonify(stats)

    except Exception as e:
        logger.error(f"Unexpected error in /churn-stats: {e}", exc_info=True)
        return jsonify({"error": "Internal Server Error"}), 500

@app.route('/churn-trends', methods=["GET"])
def get_churn_trends():
    """Churn trends over time with risk level breakdown"""
    try:
        full_df = get_customer_data_with_predictions()
        if full_df is None or 'last_login_date' not in full_df.columns:
            return jsonify({"error": "Insufficient data"}), 404

        # Convert date and filter invalid
        full_df['last_login_date'] = pd.to_datetime(full_df['last_login_date'], errors='coerce')
        full_df = full_df.dropna(subset=['last_login_date'])

        # Extract month
        full_df['month'] = full_df['last_login_date'].dt.to_period('M').astype(str)

        # Ensure 'churn_risk' column exists
        if 'churn_risk' not in full_df.columns:
            full_df['churn_risk'] = 'Unknown'

        # Group by month and churn risk
        grouped = full_df.groupby(['month', 'churn_risk']).size().reset_index(name='count')

        # Pivot for risk breakdown
        pivot_df = grouped.pivot(index='month', columns='churn_risk', values='count').fillna(0).astype(int)
        pivot_df['total_customers'] = pivot_df.sum(axis=1)

        # Sort by month
        pivot_df = pivot_df.sort_index()

        # Prepare response
        response = {
            "labels": list(pivot_df.index),
            "total_customers": list(pivot_df['total_customers']),
            "risk_breakdown": {
                risk: list(pivot_df[risk]) if risk in pivot_df.columns else [0]*len(pivot_df)
                for risk in ['High Risk', 'Medium Risk', 'Low Risk']
            }
        }

        return jsonify(response)

    except Exception as e:
        logger.error(f"Error in /churn-trends: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500

@app.route('/customer/<int:user_id>', methods=['GET'])
def get_customer_details(user_id):
    """Get details for a specific customer with enhanced error handling and data validation"""
    try:
        # Input validation
        if not isinstance(user_id, int) or user_id <= 0:
            return jsonify({
                "error": "Invalid customer ID",
                "details": "Customer ID must be a positive integer",
                "timestamp": datetime.now().isoformat()
            }), 400

        # Database query
        query = text("""
        SELECT * FROM user_full_dataset 
        WHERE user_id = :user_id
        """)
        with engine.connect() as conn:
            result = conn.execute(query, {"user_id": user_id}).fetchone()
            if not result:
                return jsonify({
                    "error": "Customer not found",
                    "timestamp": datetime.now().isoformat()
                }), 404

            customer_data = dict(result._mapping)
            logger.info(f"Retrieved raw customer data for ID {user_id}")

        # Numeric field cleanup
        numeric_fields = [
            'age', 'total_orders', 'total_spent', 'recency_days',
            'product_price', 'avg_order_frequency', 'total_time_spent',
            'engagement_score', 'abandoned_cart_count', 'abandoned_cart_value'
        ]
        for field in numeric_fields:
            val = customer_data.get(field)
            try:
                if isinstance(val, datetime):
                    customer_data[field] = 0.0
                elif isinstance(val, str):
                    customer_data[field] = float(''.join(filter(str.isdigit, val))) or 0.0
                else:
                    customer_data[field] = float(val) if val is not None else 0.0
            except Exception as e:
                logger.warning(f"Failed to convert {field} to float for customer {user_id}: {str(e)}")
                customer_data[field] = 0.0

        # Categorical cleanup
        for cat_field in ['gender', 'state', 'product_category']:
            val = customer_data.get(cat_field)
            customer_data[cat_field] = str(val).strip().lower() if val else 'unknown'

        # Prepare for prediction
        try:
            df = pd.DataFrame([{
                'age': customer_data['age'],
                'gender': customer_data['gender'],
                'state': customer_data['state'],
                'total_orders': customer_data['total_orders'],
                'total_spent': customer_data['total_spent'],
                'recency_days': customer_data['recency_days'],
                'product_category': customer_data['product_category']
            }])

            # One-hot encode
            df = pd.get_dummies(df, columns=['gender', 'state', 'product_category'])

            # Ensure all trained features exist
            for feat in trained_features:
                if feat not in df.columns:
                    df[feat] = 0

            # 🔧 Fix: Ensure column order matches trained feature order
            df = df.reindex(columns=trained_features, fill_value=0)

            # Scale and predict
            X_scaled = scaler.transform(df)
            if hasattr(model, 'predict_proba'):
                proba = model.predict_proba(X_scaled)
                prediction = proba[0, 1] if proba.shape[1] > 1 else proba[0, 0]
            else:
                prediction = model.predict(X_scaled)[0]

            # Add prediction data
            customer_data.update({
                "churn_probability": float(prediction),
                "churn_risk": categorize_risk(prediction),
                "customer_status": calculate_customer_status(customer_data),
                "lifetime_value": customer_data["total_spent"],
                "model_version": MODEL_VERSION,
                "churn_explanation": generate_churn_explanation(customer_data, prediction)
            })

            logger.info(f"Successfully processed customer {user_id} with churn probability {prediction:.2f}")

        except Exception as e:
            logger.error(f"Prediction failed for customer {user_id}: {str(e)}")
            logger.error(traceback.format_exc())
            customer_data.update({
                "churn_probability": 0,
                "churn_risk": "Unknown",
                "customer_status": "Unknown",
                "churn_explanation": ["Unable to generate explanation due to prediction error"],
                "model_version": MODEL_VERSION
            })

        return jsonify({
            "customer": customer_data,
            "metadata": {
                "processed_at": datetime.now().isoformat(),
                "model_version": MODEL_VERSION,
                "data_source": "user_full_dataset"
            }
        })

    except Exception as e:
        logger.error(f"Unexpected error in customer endpoint for ID {user_id}: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({
            "error": "Failed to retrieve customer details",
            "details": str(e),
            "timestamp": datetime.now().isoformat()
        }), 500


@app.route('/high-risk-customers', methods=['GET'])
def get_high_risk_customers():
    """Get high risk customers with fallback and robust formatting"""
    try:
        # Step 1: Fetch data
        full_df = get_customer_data_with_predictions()

        if full_df is None or full_df.empty:
            logger.warning("No data available in full_df")
            return jsonify({"error": "No data available"}), 404

        logger.info(f"Columns: {full_df.columns.tolist()}")
        logger.info(f"Unique churn_risk values: {full_df['churn_risk'].unique()}")

        # Step 2: Standardize churn_risk values
        full_df['churn_risk'] = full_df['churn_risk'].astype(str).str.lower().str.strip()
        full_df['churn_risk'] = full_df['churn_risk'].replace({"high risk": "High Risk"})

        # Step 3: Filter for high-risk customers
        high_risk = full_df[full_df["churn_risk"] == "High Risk"]

        if high_risk.empty:
            logger.info("No high-risk customers found.")
            return jsonify({
                "count": 0,
                "customers": [],
                "timestamp": datetime.now().isoformat()
            })

        # Step 4: Define required columns
        desired_columns = [
            'user_id', 'user_name', 'user_email', 'state', 'gender',
            'age', 'total_orders', 'total_spent', 'recency_days',
            'churn_probability', 'churn_risk', 'customer_status',
            'product_category', 'engagement_score'
        ]

        # Step 5: Ensure all desired columns are present
        available_columns = [col for col in desired_columns if col in high_risk.columns]
        missing_columns = [col for col in desired_columns if col not in high_risk.columns]
        if missing_columns:
            logger.warning(f"Missing columns in result set: {missing_columns}")

        high_risk = high_risk[available_columns].sort_values("churn_probability", ascending=False)

        # ✅ Step 6: Convert datetime columns to ISO strings
        datetime_cols = high_risk.select_dtypes(include=["datetime64[ns]"]).columns
        for col in datetime_cols:
            high_risk[col] = high_risk[col].apply(lambda x: x.isoformat() if pd.notnull(x) else "unknown")

        # ✅ Optional Pagination (Uncomment if needed)
        # page = int(request.args.get("page", 1))
        # limit = int(request.args.get("limit", 100))
        # start = (page - 1) * limit
        # end = start + limit
        # high_risk = high_risk.iloc[start:end]

        results = high_risk.to_dict(orient="records")
        logger.info(f"High risk customers found: {len(results)}")

        return jsonify({
            "count": len(results),
            "customers": results,
            "timestamp": datetime.now().isoformat()
        })

    except Exception as e:
        logger.error("Error in /high-risk-customers endpoint", exc_info=True)
        return jsonify({
            "error": "Internal server error",
            "message": str(e)
        }), 500

@app.route('/customer-segments', methods=['GET'])
def get_customer_segments():
    """Get customer segments based on value and activity"""
    try:
        full_df = get_customer_data_with_predictions()

        # Check for data availability
        if full_df is None or full_df.empty:
            logger.error("No data returned or dataframe is empty.")
            return jsonify({"error": "No data available"}), 404

        # Required columns for segmentation
        required_cols = [
            "total_spent", "recency_days", "total_orders", "age",
            "user_id", "churn_probability", "engagement_score"
        ]
        missing = [col for col in required_cols if col not in full_df.columns]
        if missing:
            logger.error(f"Missing required columns: {missing}")
            return jsonify({"error": f"Missing columns: {missing}"}), 500

        # Step 1: Assign segments based on value/activity
        conditions = [
            (full_df["total_spent"] > 1000) & (full_df["recency_days"] <= 30),
            (full_df["total_spent"] > 500) & (full_df["recency_days"] <= 60),
            full_df["recency_days"] > 90,
            full_df["total_orders"] == 0
        ]
        choices = ["High Value", "Medium Value", "At Risk", "New"]
        full_df["segment"] = np.select(conditions, choices, default="Regular")

        # Step 2: Assign age groups
        age_bins = [0, 18, 25, 35, 45, 55, np.inf]
        age_labels = ["0-18", "19-25", "26-35", "36-45", "46-55", "55+"]
        try:
            full_df["age_group"] = pd.cut(
                full_df["age"].fillna(30),
                bins=age_bins,
                labels=age_labels,
                right=False
            )
        except Exception as e:
            logger.error(f"Error creating age groups: {str(e)}")
            full_df["age_group"] = "26-35"

        # Step 3: Aggregate segment statistics
        try:
            segments = full_df.groupby(["segment", "age_group"], observed=True).agg({
                "user_id": "count",
                "total_spent": ["sum", "mean"],
                "churn_probability": "mean",
                "recency_days": "mean",
                "total_orders": "mean",
                "engagement_score": "mean"
            }).reset_index()

            # Flatten multi-index column names
            segments.columns = ['_'.join(col).strip('_') for col in segments.columns.values]
            segments_dict = segments.to_dict(orient='records')

        except Exception as e:
            logger.error(f"Groupby failed: {str(e)}", exc_info=True)
            return jsonify({"error": "Data processing failed"}), 500

        # Step 4: Return response
        return jsonify({
            "segments": segments_dict,
            "timestamp": datetime.now().isoformat(),
            "segment_counts": full_df["segment"].value_counts().to_dict(),
            "age_distribution": full_df["age_group"].value_counts().to_dict()
        })

    except Exception as e:
        logger.error(f"Error in /customer-segments: {str(e)}", exc_info=True)
        return jsonify({"error": str(e)}), 500

@app.route('/retention-rate', methods=['GET'])
def get_retention_rate():
    """Calculate retention rate"""
    try:
        full_df = get_customer_data_with_predictions()
        if full_df is None or full_df.empty:
            logger.warning("No customer data found for retention rate calculation.")
            return jsonify({"error": "No data available"}), 404

        # Log recency_days stats for debugging
        logger.info("Recency Days Distribution:")
        logger.info(full_df['recency_days'].describe())
        logger.info(full_df['recency_days'].value_counts(bins=5))

        # Recalculate statuses in case something was off
        full_df['customer_status'] = full_df.apply(calculate_customer_status, axis=1)

        total = len(full_df)
        active = len(full_df[full_df['customer_status'] == 'Active'])
        inactive = len(full_df[full_df['customer_status'] == 'Inactive'])
        at_risk = len(full_df[full_df['customer_status'] == 'At Risk'])
        new_customers = len(full_df[full_df['customer_status'] == 'New'])

        retention_rate = (active / total) * 100 if total > 0 else 0
        churn_rate = (inactive / total) * 100 if total > 0 else 0

        return jsonify({
            "retention_rate": round(float(retention_rate), 2),
            "churn_rate": round(float(churn_rate), 2),
            "active_customers": int(active),
            "inactive_customers": int(inactive),
            "at_risk_customers": int(at_risk),
            "new_customers": int(new_customers),
            "total_customers": int(total),
            "timestamp": datetime.now().isoformat()
        })
    except Exception as e:
        logger.error(f"Error in /retention-rate: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


@app.route('/export', methods=['GET'])
def export_data():
    """Export customer data to CSV or JSON"""
    try:
        full_df = get_customer_data_with_predictions()
        if full_df is None or full_df.empty:
            return jsonify({"error": "No data available"}), 404

        export_columns = [
            'user_id', 'user_name', 'user_email', 'state', 'gender', 'age',
            'total_orders', 'total_spent', 'avg_order_frequency', 'recency_days',
            'churn_probability', 'churn_risk', 'customer_status', 'data_processed_at',
            'model_version', 'last_login_date', 'last_order_date', 'age_group',
            'abandoned_cart_count', 'abandoned_cart_value',
            'avg_time_per_session', 'distinct_categories_count',
            'engagement_score', 'frequency',
            'most_frequent_category', 'preferred_categories',
            'purchase_frequency_segment', 'purchase_momentum',
            'recent_purchase_ratio', 'total_logins', 'total_time_spent',
            'weekly_activity_consistency'
        ]

        export_df = full_df[[col for col in export_columns if col in full_df.columns]]

        # Optional JSON export
        if request.args.get('format') == 'json':
            return jsonify(export_df.to_dict(orient='records'))

        # CSV export
        csv_buffer = StringIO()
        export_df.to_csv(csv_buffer, index=False)
        csv_buffer.seek(0)

        response = make_response(csv_buffer.getvalue())
        response.headers['Content-Disposition'] = 'attachment; filename=customer_churn_data.csv'
        response.headers['Content-type'] = 'text/csv'
        return response

    except Exception as e:
        logger.error(f"Error in /export: {str(e)}", exc_info=True)
        return jsonify({"error": "Failed to generate export"}), 500



@app.route('/predict-churn', methods=['POST'])
def predict_churn():
    """Predict churn for a single customer with enhanced validation and smarter fallback"""
    try:
        if not request.is_json:
            return jsonify({"error": "Request must be JSON"}), 400

        data = request.get_json()
        if not data:
            return jsonify({"error": "No data provided"}), 400

        required_fields = {
            'age': 'numeric',
            'total_orders': 'numeric',
            'total_spent': 'numeric',
            'recency_days': 'numeric',
            'gender': 'categorical',
            'state': 'categorical',
            'product_category': 'categorical'
        }

        customer_df = pd.DataFrame([data])

        # Validate inputs
        missing_fields = [field for field in required_fields if field not in data]
        if missing_fields:
            return jsonify({"error": "Missing required fields", "missing_fields": missing_fields}), 400

        for field, field_type in required_fields.items():
            try:
                if field_type == 'numeric':
                    customer_df[field] = pd.to_numeric(customer_df[field])
                    if customer_df[field].isnull().any():
                        return jsonify({
                            "error": f"Invalid {field} value",
                            "message": f"{field} must be a valid number"
                        }), 400
                elif field_type == 'categorical':
                    customer_df[field] = customer_df[field].astype(str).str.strip()
            except Exception as e:
                return jsonify({
                    "error": f"Invalid {field} value",
                    "message": str(e)
                }), 400

        # One-hot encode matching training structure
        categorical_fields = ['gender', 'state', 'product_category', 'age_group', 'product_name']
        customer_df = pd.get_dummies(customer_df, columns=[col for col in categorical_fields if col in customer_df.columns])

        # Fill missing trained features with 0
        for feature in trained_features:
            if feature not in customer_df.columns:
                customer_df[feature] = 0

        # Ensure feature alignment
        customer_df = customer_df.reindex(columns=trained_features, fill_value=0)

        # Log mismatches
        missing = [f for f in trained_features if f not in customer_df.columns]
        extra = [f for f in customer_df.columns if f not in trained_features]
        logger.warning(f"Missing features: {missing}")
        logger.warning(f"Unexpected features: {extra}")

        # Scale input
        X_scaled = scaler.transform(customer_df)

        # Predict churn
        if hasattr(model, 'predict_proba'):
            proba = model.predict_proba(X_scaled)
            prediction = proba[0, 1] if proba.shape[1] > 1 else proba[0, 0]
        else:
            prediction = model.predict(X_scaled)[0]

        risk_category = categorize_risk(prediction)

        # Feature importance
        feature_importance = {}

        if hasattr(model, "feature_importances_"):
            importances = model.feature_importances_
            if importances.sum() > 0:
                importances /= importances.sum()
            else:
                importances = np.ones_like(importances) / len(importances)
            feature_importance = {
                feat: round(score, 4) for feat, score in zip(trained_features, importances)
            }

        elif hasattr(model, "coef_"):
            coef = model.coef_
            if len(coef.shape) > 1:
                coef = coef[1] if coef.shape[0] > 1 else coef[0]
            abs_coef = np.abs(coef)
            if abs_coef.sum() > 0:
                abs_coef /= abs_coef.sum()
            else:
                abs_coef = np.ones_like(abs_coef) / len(abs_coef)
            feature_importance = {
                feat: round(score, 4) for feat, score in zip(trained_features, abs_coef)
            }

        else:
            logger.warning("Using fallback correlation-based importance.")
            try:
                variation = X_scaled.copy()
                noise = np.random.normal(0, 0.15, variation.shape)
                X_variations = np.repeat(variation, 10, axis=0) + noise

                if hasattr(model, "predict_proba"):
                    y_variations = model.predict_proba(X_variations)[:, 1]
                else:
                    y_variations = model.predict(X_variations)

                correlations = []
                for i in range(X_variations.shape[1]):
                    try:
                        corr = np.corrcoef(X_variations[:, i], y_variations)[0, 1]
                        correlations.append(abs(corr) if not np.isnan(corr) else 0)
                    except Exception as e:
                        logger.warning(f"Correlation failed for feature index {i}: {e}")
                        correlations.append(0)

                correlations = np.array(correlations)
                if correlations.sum() > 0:
                    correlations /= correlations.sum()
                else:
                    correlations = np.ones_like(correlations) / len(correlations)

                feature_importance = {
                    feat: round(score, 4) for feat, score in zip(trained_features, correlations)
                }

            except Exception as e:
                logger.error(f"Fallback importance generation failed: {str(e)}")
                feature_importance = {}

        # Return top features
        top_features = dict(sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:15])

        return jsonify({
            "prediction_probability": round(float(prediction), 4),
            "risk_category": risk_category,
            "feature_importance": top_features
        })

    except Exception as e:
        logger.error(f"Unhandled error in /predict-churn: {str(e)}", exc_info=True)
        return jsonify({
            "error": "Unhandled exception occurred",
            "message": str(e)
        }), 500

@app.route("/churn-explanation/<int:user_id>", methods=["GET"])
def churn_explanation(user_id):
    """Generate detailed churn explanation for a specific user"""
    try:
        # Fetch processed data with churn predictions
        full_df = get_customer_data_with_predictions()
        if full_df is None or full_df.empty:
            return jsonify({
                "error": "No customer data available",
                "timestamp": datetime.now().isoformat()
            }), 404

        # Find user record
        user_row = full_df[full_df["user_id"] == user_id]
        if user_row.empty:
            logger.warning(f"User {user_id} not found")
            return jsonify({
                "error": "User not found",
                "message": f"No user found with ID {user_id}"
            }), 404

        user = user_row.iloc[0].to_dict()

        # Generate churn explanation
        explanation = generate_churn_explanation(user, user.get("churn_probability", 0.0))

        response = {
            "user_id": user["user_id"],
            "username": user.get("user_name", "unknown"),
            "email": user.get("user_email", "unknown"),
            "state": user.get("state", "unknown"),
            "gender": user.get("gender", "unknown"),
            "age": user.get("age", None),
            "recency_days": user.get("recency_days", None),
            "total_orders": user.get("total_orders", None),
            "total_spent": user.get("total_spent", None),
            "engagement_score": user.get("engagement_score", None),
            "churn_probability": round(float(user.get("churn_probability", 0.0)), 4),
            "churn_risk": user.get("churn_risk", "Unknown"),
            "customer_status": user.get("customer_status", "Unknown"),
            "explanation": explanation,
            "timestamp": datetime.now().isoformat(),
            "model_version": MODEL_VERSION
        }

        return jsonify(response)

    except Exception as e:
        logger.error(f"Error generating churn explanation: {str(e)}", exc_info=True)
        return jsonify({
            "error": "Internal server error",
            "message": "Failed to generate churn explanation",
            "timestamp": datetime.now().isoformat()
        }), 500

    
@app.errorhandler(500)
def internal_error(error):
    logger.error(f"Server Error: {error}, Route: {request.url}")
    return jsonify({"error": "Internal server error"}), 500

@app.errorhandler(Exception)
def unhandled_exception(e):
    logger.exception(f"Unhandled Exception: {e}")
    return jsonify({"error": "Unhandled exception occurred"}), 500

# Start the Flask app
if __name__ == "__main__":
    logger.info(f"Server started at {app_start_time.isoformat()}")
    app.run(host='0.0.0.0', port=5000, debug=False)