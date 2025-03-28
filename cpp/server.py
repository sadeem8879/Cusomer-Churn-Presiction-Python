# import os
# import pickle
# import pandas as pd
# import numpy as np
# from flask import Flask, jsonify
# from flask_cors import CORS
# import logging
# from sqlalchemy import create_engine
# from dotenv import load_dotenv

# # ✅ Load Environment Variables
# load_dotenv()
# DB_URL = os.getenv("DATABASE_URL")

# if not DB_URL:
#     logging.error("❌ DATABASE_URL is missing! Check your .env file.")
#     raise ValueError("Missing DATABASE_URL")

# # ✅ Flask App Setup
# app = Flask(__name__)
# CORS(app)
# logging.basicConfig(level=logging.DEBUG)

# # ✅ Load Model & Preprocessing Files
# BASE_DIR = os.path.dirname(__file__)
# MODEL_PATH = os.path.join(BASE_DIR, "model.pkl")
# SCALER_PATH = os.path.join(BASE_DIR, "scaler.pkl")
# FEATURES_PATH = os.path.join(BASE_DIR, "trained_features.pkl")

# # ✅ Ensure Required Files Exist
# for path in [MODEL_PATH, SCALER_PATH, FEATURES_PATH]:
#     if not os.path.exists(path):
#         logging.error(f"❌ Missing required file: {path}")
#         raise FileNotFoundError(f"❌ Model, Scaler, or Features file is missing!")

# # ✅ Load Model, Scaler & Features
# with open(MODEL_PATH, "rb") as model_file:
#     model = pickle.load(model_file)

# with open(SCALER_PATH, "rb") as scaler_file:
#     scaler = pickle.load(scaler_file)

# with open(FEATURES_PATH, "rb") as features_file:
#     trained_features = pickle.load(features_file)

# # ✅ Function to Categorize Risk Levels based on probabilities
# def categorize_risk(probability):
#     if probability >= 0.7:
#         return 'High Risk'
#     elif 0.3 <= probability < 0.7:
#         return 'Medium Risk'
#     else:
#         return 'Low Risk'

# # ✅ API: Predict Churn Statistics
# @app.route("/predict-churn-stats", methods=["GET"])
# def predict_churn_statistics():
#     try:
#         # ✅ Connect to Database
#         engine = create_engine(DB_URL)
        
#         # ✅ Fetch Data
#         query = """
#         SELECT user_id, gender, state, age, recency_days, total_orders, total_logins, avg_time_per_session, abandoned_cart_count
#         FROM user_full_dataset;
#         """
#         df = pd.read_sql(query, engine)

#         if df.empty:
#             return jsonify({
#                 "churn_by_gender": [],
#                 "churn_by_state": [],
#                 "churn_by_age": [],
#                 "risk_levels": [],
#                 "active_vs_churned": []
#             })

#         # ✅ Feature Engineering
#         df["frequency"] = df["total_orders"] / ((df["recency_days"].replace(0, np.nan) / 30) + 1)
#         df["frequency"] = df["frequency"].fillna(0)

#         df.drop(columns=["user_id"], errors="ignore", inplace=True)

#         # ✅ Encode Categorical Variables
#         df = pd.get_dummies(df, columns=["gender", "state"], drop_first=True)

#         # ✅ Align Columns with Trained Model
#         for col in trained_features:
#             if col not in df.columns:
#                 df[col] = 0  # Add missing categorical features

#         df = df[trained_features]  # Ensure correct column order

#         # ✅ Scale Features
#         X_scaled = scaler.transform(df)

#         # ✅ Predict Churn Probabilities
#         df["churn_probability"] = model.predict_proba(X_scaled)[:, 1]

#         # ✅ Predict Churn (for active/churned count)
#         df["churn"] = (df["churn_probability"] >= 0.5).astype(int)

#         # ✅ Categorize Risk Levels
#         df['risk_level'] = df['churn_probability'].apply(categorize_risk)

#         # ✅ Aggregate Churn Data
#         churn_by_gender = df.groupby("gender")["churn"].sum().reset_index()
#         churn_by_gender.rename(columns={"churn": "customer_count"}, inplace=True)

#         churn_by_state = df.groupby("state")["churn"].sum().reset_index()
#         churn_by_state.rename(columns={"churn": "customer_count"}, inplace=True)

#         df["age_group"] = pd.cut(df["age"], bins=[0, 17, 24, 34, 44, 54, np.inf], 
#                                  labels=["Under 18", "18-24", "25-34", "35-44", "45-54", "55+"], right=False)

#         churn_by_age = df.groupby("age_group")["churn"].sum().reset_index()
#         churn_by_age.rename(columns={"churn": "customer_count"}, inplace=True)

#         risk_levels = df['risk_level'].value_counts().reset_index()
#         risk_levels.columns = ['risk_level', 'count']

#         active_vs_churned = df["churn"].value_counts().reset_index()
#         active_vs_churned.columns = ['churn_status', 'count']
#         active_vs_churned["churn_status"] = active_vs_churned["churn_status"].map({0: "Active", 1: "Churned"})

#         return jsonify({
#             "churn_by_gender": churn_by_gender.to_dict(orient="records"),
#             "churn_by_state": churn_by_state.to_dict(orient="records"),
#             "churn_by_age": churn_by_age.to_dict(orient="records"),
#             "risk_levels": risk_levels.to_dict(orient="records"),
#             "active_vs_churned": active_vs_churned.to_dict(orient="records")
#         })

#     except Exception as e:
#         logging.error(f"❌ Error predicting churn statistics: {e}")
#         return jsonify({"error": str(e)}), 500

# # ✅ Run Flask Server
# if __name__ == "__main__":
#     app.run(debug=True)

# ml.py (No changes needed, as it's already well-structured)







# # server.py
# import os
# import pickle
# import pandas as pd
# import numpy as np
# from flask import Flask, jsonify, request
# from flask_cors import CORS
# import logging
# from sqlalchemy import create_engine
# from dotenv import load_dotenv
# from sqlalchemy import text

# # ✅ Load environment variables
# load_dotenv()
# DB_URL = os.getenv("DATABASE_URL")

# if not DB_URL:
#     logging.error("❌ DATABASE_URL is missing! Check your .env file.")
#     raise ValueError("Missing DATABASE_URL")

# # ✅ Flask app setup
# app = Flask(__name__)
# CORS(app)
# logging.basicConfig(level=logging.DEBUG)

# # ✅ Load model, scaler, and features
# BASE_DIR = os.path.dirname(__file__)
# MODEL_PATH = os.path.join(BASE_DIR, "model.pkl")
# SCALER_PATH = os.path.join(BASE_DIR, "scaler.pkl")
# FEATURES_PATH = os.path.join(BASE_DIR, "trained_features.pkl")

# # Ensure required files exist
# for path in [MODEL_PATH, SCALER_PATH, FEATURES_PATH]:
#     if not os.path.exists(path):
#         logging.error(f"❌ Missing required file: {path}")
#         raise FileNotFoundError(f"❌ Model, Scaler, or Features file is missing!")

# # Load model, scaler, and features
# try:
#     with open(MODEL_PATH, "rb") as f:
#         model = pickle.load(f)
#     with open(SCALER_PATH, "rb") as f:
#         scaler = pickle.load(f)
#     with open(FEATURES_PATH, "rb") as f:
#         trained_features = pickle.load(f)
#     logging.info("✅ Model, scaler, and features loaded successfully!")
# except Exception as e:
#     logging.error(f"❌ Error loading model, scaler, or features: {e}")
#     exit()

# # ✅ Database connection
# try:
#     engine = create_engine(DB_URL)
#     with engine.connect() as conn:
#         result = conn.execute(text("SELECT 1")).fetchone()  # Test connection
#         logging.info(f"✅ Database Test Query Result: {result[0]}")
#     logging.info("✅ Connected to database!")
# except Exception as e:
#     logging.error(f"❌ Database connection failed: {e}")
#     exit()

# # ✅ Function to categorize risk levels
# def categorize_risk(probability):
#     if probability >= 0.7:
#         return 'High Risk'
#     elif 0.3 <= probability < 0.7:
#         return 'Medium Risk'
#     else:
#         return 'Low Risk'

# # ✅ Endpoint: Predict churn for a batch of customers
# @app.route('/predict-churn-batch', methods=['POST'])
# def predict_churn_batch():
#     try:
#         logging.info("✅ /predict-churn-batch endpoint hit!")
#         data = request.get_json()

#         if not data or 'customers' not in data:
#             logging.error("❌ Invalid request. Missing 'customers' key in JSON payload.")
#             return jsonify({"error": "Invalid request. Provide a JSON payload with a 'customers' list."}), 400

#         customers = data['customers']

#         if not customers:
#             logging.error("❌ No customers provided in the 'customers' list.")
#             return jsonify({"error": "No customers provided in the 'customers' list."}), 400

#         # Convert customers to DataFrame
#         try:
#             df = pd.DataFrame(customers)
#         except ValueError as e:
#             logging.error(f"❌ Error converting customers data to DataFrame: {e}")
#             return jsonify({"error": "Invalid customer data format. Ensure data is a list of dictionaries."}), 400

#         # Log input data for debugging
#         logging.info("Input Data (Before Preprocessing):\n%s", df.head().to_string())
#         logging.info(f"Shape of input data before alignment: {df.shape}")

#         # Ensure all trained features are present
#         for col in trained_features:
#             if col not in df.columns:
#                 df[col] = 0  # Add missing columns with default value

#         # Align columns with trained features
#         df = df[trained_features]

#         # Log aligned data for debugging
#         logging.info("Input Data (After Alignment):\n%s", df.head().to_string())
#         logging.info(f"Shape of input data after alignment: {df.shape}")
#         logging.info(f"Trained features: {trained_features}")
#         logging.info(f"Data types of input data:\n{df.dtypes}")
#         logging.info(f"Missing values in input data:\n{df.isnull().sum()}")

#         # Scale features
#         try:
#             X_scaled = scaler.transform(df)
#             logging.info(f"Shape of scaled data: {X_scaled.shape}")
#         except ValueError as e:
#             logging.error(f"❌ Error scaling features: {e}")
#             return jsonify({"error": "Error scaling features. Ensure input data matches training data."}), 500

#         # Predict churn probabilities
#         try:
#             predictions = model.predict_proba(X_scaled)[:, 1]
#             logging.info(f"Shape of model predictions: {predictions.shape}")
#             if len(predictions) == 0:
#                 return jsonify({"predictions": []}), 200
#             df["churn_probability"] = predictions
#         except Exception as e:
#             logging.error(f"❌ Error during prediction: {e}", exc_info=True)
#             return jsonify({"error": f"Error during prediction: {e}"}), 500

#         # Return predictions
#         return jsonify({"predictions": df["churn_probability"].tolist()}), 200

#     except Exception as e:
#         logging.error(f"❌ Error in /predict-churn-batch: {e}", exc_info=True)
#         return jsonify({"error": f"Internal server error: {e}"}), 500
# # ✅ Endpoint: Get churn trends
# @app.route('/churn-trends', methods=['GET'])
# def get_churn_trends():
#     try:
#         # Fetch data from the database
#         query = text("SELECT * FROM user_full_dataset") #Use text to prevent SQL injection
#         df = pd.read_sql(query, engine)

#         if df.empty:
#             return jsonify({"error": "No data found in the database"}), 404

#         # Feature engineering
#         df["frequency"] = df["total_orders"] / ((df["recency_days"].replace(0, np.nan) / 30) + 1)
#         df["frequency"] = df["frequency"].fillna(0)

#         # Align columns with trained features
#         for col in trained_features:
#             if col not in df.columns:
#                 df[col] = 0  # Add missing columns with default value

#         df = df[trained_features]  # Ensure correct column order

#         # Scale features
#         X_scaled = scaler.transform(df)

#         # Predict churn probabilities
#         df["churn_probability"] = model.predict_proba(X_scaled)[:, 1]

#         # Group by month for trends
#         df["month"] = pd.to_datetime(df["last_login_date"]).dt.to_period("M")
#         trends = df.groupby("month")["churn_probability"].mean().reset_index()

#         # Convert month to string for JSON serialization
#         trends['month'] = trends['month'].astype(str)
#         return jsonify(trends.to_dict(orient="records"))
#     except Exception as e:
#         logging.error(f"❌ Error in /churn-trends: {e}")
#         return jsonify({"error": f"Internal server error: {e}"}), 500

# # ✅ Endpoint: Get churn by state
# @app.route('/churned-state', methods=['GET'])
# def get_churn_by_state():
#     try:
#         # Fetch data from the database
#         query = text("SELECT * FROM user_full_dataset")
#         df = pd.read_sql(query, engine)

#         if df.empty:
#             return jsonify({"error": "No data found in the database"}), 404

#         # Feature engineering
#         df["frequency"] = df["total_orders"] / ((df["recency_days"].replace(0, np.nan) / 30) + 1)
#         df["frequency"] = df["frequency"].fillna(0)

#         # Align columns with trained features
#         for col in trained_features:
#             if col not in df.columns:
#                 df[col] = 0  # Add missing columns with default value

#         df = df[trained_features]  # Ensure correct column order

#         # Scale features
#         X_scaled = scaler.transform(df)

#         # Predict churn probabilities
#         df["churn_probability"] = model.predict_proba(X_scaled)[:, 1]

#         # Group by state
#         churn_by_state = df.groupby("state")["churn_probability"].mean().reset_index()

#         return jsonify(churn_by_state.to_dict(orient="records"))
#     except Exception as e:
#         logging.error(f"❌ Error in /churned-state: {e}")
#         return jsonify({"error": f"Internal server error: {e}"}), 500

# # ✅ Endpoint: Get churn by gender
# @app.route('/churned-gender', methods=['GET'])
# def get_churn_by_gender():
#     try:
#         # Fetch data from the database
#         query = text("SELECT * FROM user_full_dataset")
#         df = pd.read_sql(query, engine)

#         if df.empty:
#             return jsonify({"error": "No data found in the database"}), 404

#         # Feature engineering
#         df["frequency"] = df["total_orders"] / ((df["recency_days"].replace(0, np.nan) / 30) + 1)
#         df["frequency"] = df["frequency"].fillna(0)

#         # Align columns with trained features
#         for col in trained_features:
#             if col not in df.columns:
#                 df[col] = 0  # Add missing columns with default value

#         df = df[trained_features]  # Ensure correct column order

#         # Scale features
#         X_scaled = scaler.transform(df)

#         # Predict churn probabilities
#         df["churn_probability"] = model.predict_proba(X_scaled)[:, 1]

#         # Group by gender
#         churn_by_gender = df.groupby("gender")["churn_probability"].mean().reset_index()

#         return jsonify(churn_by_gender.to_dict(orient="records"))
#     except Exception as e:
#         logging.error(f"❌ Error in /churned-gender: {e}")
#         return jsonify({"error": f"Internal server error: {e}"}), 500

# # ✅ Endpoint: Get churn by age
# @app.route('/churned-age', methods=['GET'])
# def get_churn_by_age():
#     try:
#         # Fetch data from the database
#         query = text("SELECT * FROM user_full_dataset")
#         df = pd.read_sql(query, engine)

#         if df.empty:
#             return jsonify({"error": "No data found in the database"}), 404

#         # Feature engineering
#         df["frequency"] = df["total_orders"] / ((df["recency_days"].replace(0, np.nan) / 30) + 1)
#         df["frequency"] = df["frequency"].fillna(0)

#         # Align columns with trained features
#         for col in trained_features:
#             if col not in df.columns:
#                 df[col] = 0  # Add missing columns with default value

#         df = df[trained_features]  # Ensure correct column order

#         # Scale features
#         X_scaled = scaler.transform(df)

#         # Predict churn probabilities
#         df["churn_probability"] = model.predict_proba(X_scaled)[:, 1]

#         # Group by age
#         df["age_group"] = pd.cut(df["age"], bins=[0, 18, 25, 35, 45, 55, np.inf],
#                                     labels=["0-18", "19-25", "26-35", "36-45", "46-55", "55+"], right=False)
#         churn_by_age = df.groupby("age_group")["churn_probability"].mean().reset_index()

#         return jsonify(churn_by_age.to_dict(orient="records"))
#     except Exception as e:
#         logging.error(f"❌ Error in /churned-age: {e}")
#         return jsonify({"error": f"Internal server error: {e}"}), 500

# # ✅ Endpoint: Get high-risk customers
# @app.route('/high-risk-customers', methods=['GET'])
# def get_high_risk_customers():
#     try:
#         # Fetch data from the database
#         query = text("SELECT * FROM user_full_dataset")
#         df = pd.read_sql(query, engine)

#         if df.empty:
#             return jsonify({"error": "No data found"}), 404

#         # Feature engineering
#         df["frequency"] = df["total_orders"] / ((df["recency_days"].replace(0, np.nan) / 30) + 1)
#         df["frequency"] = df["frequency"].fillna(0)

#         # Align columns with trained features
#         for col in trained_features:
#             if col not in df.columns:
#                 df[col] = 0  # Add missing columns with default value

#         df = df[trained_features]  # Ensure correct column order

#         # Scale features
#         X_scaled = scaler.transform(df)

#         # Predict churn probabilities
#         df["churn_probability"] = model.predict_proba(X_scaled)[:, 1]

#         # Filter high-risk customers
#         high_risk_customers = df[df["churn_probability"] >= 0.7]

#         #  Handle the case where there are no high-risk customers
#         if high_risk_customers.empty:
#             return jsonify({"high_risk_customers": []}), 200
#         else:
#             return jsonify(high_risk_customers.to_dict(orient="records")), 200
#     except Exception as e:
#         logging.error(f"❌ Error in /high-risk-customers: {e}")
#         return jsonify({"error": f"Internal server error: {e}"}), 500



# # ✅ Run Flask server
# if __name__ == "__main__":
#     app.run(debug=True, port=5000)


















# import os
# import pickle
# import logging
# import numpy as np
# import pandas as pd
# from flask import Flask, jsonify, request
# from flask_cors import CORS
# from sqlalchemy import create_engine, text
# from sklearn.preprocessing import StandardScaler
# from dotenv import load_dotenv

# # ✅ Load environment variables
# load_dotenv()
# DB_URL = os.getenv("DATABASE_URL")

# if not DB_URL:
#     logging.error("❌ DATABASE_URL is missing! Check your .env file.")
#     raise ValueError("Missing DATABASE_URL")

# # ✅ Flask app setup
# app = Flask(__name__)
# CORS(app)
# logging.basicConfig(level=logging.DEBUG)

# # ✅ Load model, scaler, and features
# BASE_DIR = os.path.dirname(__file__)
# MODEL_PATH = os.path.join(BASE_DIR, "model.pkl")
# SCALER_PATH = os.path.join(BASE_DIR, "scaler.pkl")
# FEATURES_PATH = os.path.join(BASE_DIR, "trained_features.pkl")

# # Ensure required files exist
# for path in [MODEL_PATH, SCALER_PATH, FEATURES_PATH]:
#     if not os.path.exists(path):
#         logging.error(f"❌ Missing required file: {path}")
#         raise FileNotFoundError(f"❌ Model, Scaler, or Features file is missing!")

# # Load model, scaler, and features
# try:
#     with open(MODEL_PATH, "rb") as f:
#         model = pickle.load(f)
#     with open(SCALER_PATH, "rb") as f:
#         scaler = pickle.load(f)
#     with open(FEATURES_PATH, "rb") as f:
#         trained_features = pickle.load(f)
#     logging.info("✅ Model, scaler, and features loaded successfully!")
# except Exception as e:
#     logging.error(f"❌ Error loading model, scaler, or features: {e}")
#     exit()

# # ✅ Database connection
# try:
#     engine = create_engine(DB_URL)
#     with engine.connect() as conn:
#         result = conn.execute(text("SELECT 1")).fetchone()  # Test connection
#         logging.info(f"✅ Database Test Query Result: {result[0]}")
#     logging.info("✅ Connected to database!")
# except Exception as e:
#     logging.error(f"❌ Database connection failed: {e}")
#     exit()

# # ✅ Function to categorize risk levels
# def categorize_risk(probability):
#     if probability >= 0.7:
#         return 'High Risk'
#     elif 0.3 <= probability < 0.7:
#         return 'Medium Risk'
#     else:
#         return 'Low Risk'

# # ✅ Endpoint: Predict churn for a batch of customers
# @app.route('/predict-churn-batch', methods=['POST'])
# def predict_churn_batch():
#     try:
#         logging.info("✅ /predict-churn-batch endpoint hit!")
#         data = request.get_json()

#         if not data or 'customers' not in data:
#             logging.error("❌ Invalid request. Missing 'customers' key in JSON payload.")
#             return jsonify({"error": "Invalid request. Provide a JSON payload with a 'customers' list."}), 400

#         customers = data['customers']

#         if not customers:
#             logging.error("❌ No customers provided in the 'customers' list.")
#             return jsonify({"error": "No customers provided in the 'customers' list."}), 400

#         # Convert customers to DataFrame
#         try:
#             df = pd.DataFrame(customers)
#         except ValueError as e:
#             logging.error(f"❌ Error converting customers data to DataFrame: {e}")
#             return jsonify({"error": "Invalid customer data format. Ensure data is a list of dictionaries."}), 400

#         # Log input data for debugging
#         logging.info("Input Data (Before Preprocessing):\n%s", df.head().to_string())
#         logging.info(f"Shape of input data before alignment: {df.shape}")

#         # Ensure all trained features are present
#         for col in trained_features:
#             if col not in df.columns:
#                 df[col] = 0  # Add missing columns with default value

#         # Align columns with trained features
#         df = df[trained_features]

#         # Log aligned data for debugging
#         logging.info("Input Data (After Alignment):\n%s", df.head().to_string())
#         logging.info(f"Shape of input data after alignment: {df.shape}")
#         logging.info(f"Trained features: {trained_features}")
#         logging.info(f"Data types of input data:\n{df.dtypes}")
#         logging.info(f"Missing values in input data:\n{df.isnull().sum()}")

#         # Scale features
#         try:
#             X_scaled = scaler.transform(df)
#             logging.info(f"Shape of scaled data: {X_scaled.shape}")
#         except ValueError as e:
#             logging.error(f"❌ Error scaling features: {e}")
#             return jsonify({"error": "Error scaling features. Ensure input data matches training data."}), 500

#         # Predict churn probabilities
#         try:
#             proba = model.predict_proba(X_scaled)
#             logging.info(f"Shape of predict_proba output: {proba.shape}")

#             # Handle binary and multi-class cases
#             if proba.shape[1] == 1:
#                 predictions = proba[:, 0]  # Binary classifier with only one class probability
#             else:
#                 predictions = proba[:, 1]  # Binary classifier with two class probabilities

#             logging.info(f"Shape of model predictions: {predictions.shape}")
#             if len(predictions) == 0:
#                 return jsonify({"predictions": []}), 200
#             df["churn_probability"] = predictions
#         except Exception as e:
#             logging.error(f"❌ Error during prediction: {e}", exc_info=True)
#             return jsonify({"error": f"Error during prediction: {e}"}), 500

#         # Return predictions
#         return jsonify({"predictions": df["churn_probability"].tolist()}), 200

#     except Exception as e:
#         logging.error(f"❌ Error in /predict-churn-batch: {e}", exc_info=True)
#         return jsonify({"error": f"Internal server error: {e}"}), 500
    
# # ✅ Endpoint: Get churn trends
# @app.route('/churn-trends', methods=['GET'])
# def get_churn_trends():
#     try:
#         logging.info("✅ /churn-trends endpoint hit!")
        
#         # 1. Fetch all data including last_login_date
#         query = text("""
#             SELECT *, 
#                    EXTRACT(DAY FROM (CURRENT_DATE - last_login_date)) AS recency_days
#             FROM user_full_dataset
#             WHERE last_login_date IS NOT NULL
#         """)
#         full_df = pd.read_sql(query, engine)

#         if full_df.empty:
#             logging.warning("⚠️ No data found in the database")
#             return jsonify({"error": "No data found in the database"}), 404

#         # 2. Store the dates separately before feature processing
#         date_info = full_df[['user_id', 'last_login_date']].copy()
        
#         # 3. Feature engineering
#         full_df["frequency"] = full_df["total_orders"] / (full_df["recency_days"].replace(0, np.nan) / 30 + 1)
#         full_df["frequency"] = full_df["frequency"].fillna(0)

#         # 4. Ensure all trained features are present
#         for col in trained_features:
#             if col not in full_df.columns:
#                 full_df[col] = 0
#                 logging.info(f"Added missing column: {col}")

#         # 5. Create working dataframe with just the features
#         df = full_df[trained_features].copy()

#         # 6. Scale features
#         try:
#             X_scaled = scaler.transform(df)
#             logging.info(f"Scaled data shape: {X_scaled.shape}")
#         except Exception as e:
#             logging.error(f"❌ Error scaling features: {e}")
#             return jsonify({"error": "Error scaling features"}), 500

#         # 7. Predict churn probabilities - FIXED VERSION
#         try:
#             proba = model.predict_proba(X_scaled)
#             logging.info(f"Predict_proba shape: {proba.shape}")

#             # Handle both binary and single-column cases
#             if proba.shape[1] == 1:
#                 # Single column means it's either:
#                 # - Binary classifier returning only positive class probabilities
#                 # - Regression model returning single value
#                 predictions = proba[:, 0]  # Use the only column available
#             else:
#                 # Standard binary classifier with two columns
#                 predictions = proba[:, 1]  # Use positive class probabilities

#             full_df["churn_probability"] = predictions
#             logging.info("Predictions added successfully")
#         except Exception as e:
#             logging.error(f"❌ Prediction error: {e}")
#             return jsonify({"error": "Prediction failed"}), 500

#         # 8. Merge back the date information
#         result_df = full_df.merge(date_info, on='user_id', how='left')

#         # 9. Group by month
#         try:
#             result_df["month"] = pd.to_datetime(result_df["last_login_date"]).dt.to_period("M")
#             trends = result_df.groupby("month")["churn_probability"].mean().reset_index()
#             trends['month'] = trends['month'].astype(str)
            
#             logging.info(f"Trends data:\n{trends.head()}")
#             return jsonify(trends.to_dict(orient="records"))
#         except Exception as e:
#             logging.error(f"❌ Error grouping by month: {e}")
#             return jsonify({"error": "Could not generate trends"}), 500

#     except Exception as e:
#         logging.error(f"❌ Error in /churn-trends: {e}", exc_info=True)
#         return jsonify({"error": f"Internal server error: {str(e)}"}), 500
# # ✅ Endpoint: Get churn by state
# @app.route('/churned-state', methods=['GET'])
# def get_churn_by_state():
#     try:
#         # Fetch data from the database
#         query = text("SELECT * FROM user_full_dataset")
#         df = pd.read_sql(query, engine)

#         if df.empty:
#             return jsonify({"error": "No data found in the database"}), 404

#         # Feature engineering
#         df["frequency"] = df["total_orders"] / (df["recency_days"].replace(0, np.nan) / 30 + 1)
#         df["frequency"] = df["frequency"].fillna(0)

#         # Align columns with trained features
#         for col in trained_features:
#             if col not in df.columns:
#                 df[col] = 0  # Add missing columns with default value

#         df = df[trained_features]  # Ensure correct column order

#         # Scale features
#         X_scaled = scaler.transform(df)

#         # Predict churn probabilities
#         df["churn_probability"] = model.predict_proba(X_scaled)[:, 1]

#         # Group by state
#         churn_by_state = df.groupby("state")["churn_probability"].mean().reset_index()

#         return jsonify(churn_by_state.to_dict(orient="records"))
#     except Exception as e:
#         logging.error(f"❌ Error in /churned-state: {e}")
#         return jsonify({"error": f"Internal server error: {e}"}), 500

# # ✅ Endpoint: Get churn by gender
# @app.route('/churned-gender', methods=['GET'])
# def get_churn_by_gender():
#     try:
#         # Fetch data from the database
#         query = text("SELECT * FROM user_full_dataset")
#         df = pd.read_sql(query, engine)

#         if df.empty:
#             return jsonify({"error": "No data found in the database"}), 404

#         # Feature engineering
#         df["frequency"] = df["total_orders"] / (df["recency_days"].replace(0, np.nan) / 30 + 1)
#         df["frequency"] = df["frequency"].fillna(0)

#         # Align columns with trained features
#         for col in trained_features:
#             if col not in df.columns:
#                 df[col] = 0  # Add missing columns with default value

#         df = df[trained_features]  # Ensure correct column order

#         # Scale features
#         X_scaled = scaler.transform(df)

#         # Predict churn probabilities
#         df["churn_probability"] = model.predict_proba(X_scaled)[:, 1]

#         # Group by gender
#         churn_by_gender = df.groupby("gender")["churn_probability"].mean().reset_index()

#         return jsonify(churn_by_gender.to_dict(orient="records"))
#     except Exception as e:
#         logging.error(f"❌ Error in /churned-gender: {e}")
#         return jsonify({"error": f"Internal server error: {e}"}), 500

# # ✅ Endpoint: Get churn by age
# @app.route('/churned-age', methods=['GET'])
# def get_churn_by_age():
#     try:
#         # Fetch data from the database
#         query = text("SELECT * FROM user_full_dataset")
#         df = pd.read_sql(query, engine)

#         if df.empty:
#             return jsonify({"error": "No data found in the database"}), 404

#         # Feature engineering
#         df["frequency"] = df["total_orders"] / (df["recency_days"].replace(0, np.nan) / 30 + 1)
#         df["frequency"] = df["frequency"].fillna(0)

#         # Align columns with trained features
#         for col in trained_features:
#             if col not in df.columns:
#                 df[col] = 0  # Add missing columns with default value

#         df = df[trained_features]  # Ensure correct column order

#         # Scale features
#         X_scaled = scaler.transform(df)

#         # Predict churn probabilities
#         df["churn_probability"] = model.predict_proba(X_scaled)[:, 1]

#         # Group by age
#         df["age_group"] = pd.cut(df["age"], bins=[0, 18, 25, 35, 45, 55, np.inf],
#                                     labels=["0-18", "19-25", "26-35", "36-45", "46-55", "55+"], right=False)
#         churn_by_age = df.groupby("age_group")["churn_probability"].mean().reset_index()

#         return jsonify(churn_by_age.to_dict(orient="records"))
#     except Exception as e:
#         logging.error(f"❌ Error in /churned-age: {e}")
#         return jsonify({"error": f"Internal server error: {e}"}), 500

# # ✅ Endpoint: Get high-risk customers
# @app.route('/high-risk-customers', methods=['GET'])
# def get_high_risk_customers():
#     try:
#         # Fetch data from the database
#         query = text("SELECT * FROM user_full_dataset")
#         df = pd.read_sql(query, engine)

#         if df.empty:
#             return jsonify({"error": "No data found"}), 404

#         # Feature engineering
#         df["frequency"] = df["total_orders"] / (df["recency_days"].replace(0, np.nan) / 30 + 1)
#         df["frequency"] = df["frequency"].fillna(0)

#         # Align columns with trained features
#         for col in trained_features:
#             if col not in df.columns:
#                 df[col] = 0  # Add missing columns with default value

#         df = df[trained_features]  # Ensure correct column order

#         # Scale features
#         X_scaled = scaler.transform(df)

#         # Predict churn probabilities
#         df["churn_probability"] = model.predict_proba(X_scaled)[:, 1]

#         # Filter high-risk customers
#         high_risk_customers = df[df["churn_probability"] >= 0.7]

#         # Handle the case where there are no high-risk customers
#         if high_risk_customers.empty:
#             return jsonify({"high_risk_customers": []}), 200
#         else:
#             return jsonify(high_risk_customers.to_dict(orient="records")), 200
#     except Exception as e:
#         logging.error(f"❌ Error in /high-risk-customers: {e}")
#         return jsonify({"error": f"Internal server error: {e}"}), 500

# # ✅ Run Flask server
# if __name__ == "__main__":
#     app.run(debug=True, port=5000)

# import os
# import pickle
# import logging
# import numpy as np
# import pandas as pd
# from flask import Flask, jsonify, request
# from flask_cors import CORS
# from sqlalchemy import create_engine, text
# from sklearn.preprocessing import StandardScaler
# from dotenv import load_dotenv

# # ✅ Load environment variables
# load_dotenv()
# DB_URL = os.getenv("DATABASE_URL")

# if not DB_URL:
#     logging.error("❌ DATABASE_URL is missing! Check your .env file.")
#     raise ValueError("Missing DATABASE_URL")

# # ✅ Flask app setup
# app = Flask(__name__)
# CORS(app)
# logging.basicConfig(level=logging.DEBUG)

# # ✅ Load model, scaler, and features
# BASE_DIR = os.path.dirname(__file__)
# MODEL_PATH = os.path.join(BASE_DIR, "model.pkl")
# SCALER_PATH = os.path.join(BASE_DIR, "scaler.pkl")
# FEATURES_PATH = os.path.join(BASE_DIR, "trained_features.pkl")

# # Ensure required files exist
# for path in [MODEL_PATH, SCALER_PATH, FEATURES_PATH]:
#     if not os.path.exists(path):
#         logging.error(f"❌ Missing required file: {path}")
#         raise FileNotFoundError(f"❌ Model, Scaler, or Features file is missing!")

# # Load model, scaler, and features
# try:
#     with open(MODEL_PATH, "rb") as f:
#         model = pickle.load(f)
#     with open(SCALER_PATH, "rb") as f:
#         scaler = pickle.load(f)
#     with open(FEATURES_PATH, "rb") as f:
#         trained_features = pickle.load(f)
#     logging.info("✅ Model, scaler, and features loaded successfully!")
# except Exception as e:
#     logging.error(f"❌ Error loading model, scaler, or features: {e}")
#     exit()

# # ✅ Database connection
# try:
#     engine = create_engine(DB_URL)
#     with engine.connect() as conn:
#         result = conn.execute(text("SELECT 1")).fetchone()  # Test connection
#         logging.info(f"✅ Database Test Query Result: {result[0]}")
#     logging.info("✅ Connected to database!")
# except Exception as e:
#     logging.error(f"❌ Database connection failed: {e}")
#     exit()

# # ✅ Function to categorize risk levels
# def categorize_risk(probability):
#     if probability >= 0.7:
#         return 'High Risk'
#     elif 0.3 <= probability < 0.7:
#         return 'Medium Risk'
#     else:
#         return 'Low Risk'

# # ✅ Endpoint: Predict churn for a batch of customers
# @app.route('/predict-churn-batch', methods=['POST'])
# def predict_churn_batch():
#     try:
#         logging.info("✅ /predict-churn-batch endpoint hit!")
#         data = request.get_json()

#         if not data or 'customers' not in data:
#             logging.error("❌ Invalid request. Missing 'customers' key in JSON payload.")
#             return jsonify({"error": "Invalid request. Provide a JSON payload with a 'customers' list."}), 400

#         customers = data['customers']

#         if not customers:
#             logging.error("❌ No customers provided in the 'customers' list.")
#             return jsonify({"error": "No customers provided in the 'customers' list."}), 400

#         # Convert customers to DataFrame
#         try:
#             df = pd.DataFrame(customers)
#         except ValueError as e:
#             logging.error(f"❌ Error converting customers data to DataFrame: {e}")
#             return jsonify({"error": "Invalid customer data format. Ensure data is a list of dictionaries."}), 400

#         # Log input data for debugging
#         logging.info("Input Data (Before Preprocessing):\n%s", df.head().to_string())
#         logging.info(f"Shape of input data before alignment: {df.shape}")

#         # Ensure all trained features are present
#         for col in trained_features:
#             if col not in df.columns:
#                 df[col] = 0  # Add missing columns with default value

#         # Align columns with trained features
#         df = df[trained_features]

#         # Log aligned data for debugging
#         logging.info("Input Data (After Alignment):\n%s", df.head().to_string())
#         logging.info(f"Shape of input data after alignment: {df.shape}")
#         logging.info(f"Trained features: {trained_features}")
#         logging.info(f"Data types of input data:\n{df.dtypes}")
#         logging.info(f"Missing values in input data:\n{df.isnull().sum()}")

#         # Scale features
#         try:
#             X_scaled = scaler.transform(df)
#             logging.info(f"Shape of scaled data: {X_scaled.shape}")
#         except ValueError as e:
#             logging.error(f"❌ Error scaling features: {e}")
#             return jsonify({"error": "Error scaling features. Ensure input data matches training data."}), 500

#         # Predict churn probabilities
#         try:
#             proba = model.predict_proba(X_scaled)
#             logging.info(f"Shape of predict_proba output: {proba.shape}")

#             # Handle binary and multi-class cases
#             if proba.shape[1] == 1:
#                 predictions = proba[:, 0]  # Binary classifier with only one class probability
#             else:
#                 predictions = proba[:, 1]  # Binary classifier with two class probabilities

#             logging.info(f"Shape of model predictions: {predictions.shape}")
#             if len(predictions) == 0:
#                 return jsonify({"predictions": []}), 200
#             df["churn_probability"] = predictions
#         except Exception as e:
#             logging.error(f"❌ Error during prediction: {e}", exc_info=True)
#             return jsonify({"error": f"Error during prediction: {e}"}), 500

#         # Return predictions
#         return jsonify({"predictions": df["churn_probability"].tolist()}), 200

#     except Exception as e:
#         logging.error(f"❌ Error in /predict-churn-batch: {e}", exc_info=True)
#         return jsonify({"error": f"Internal server error: {e}"}), 500
    
# # ✅ Endpoint: Get churn trends
# @app.route('/churn-trends', methods=['GET'])
# def get_churn_trends():
#     try:
#         logging.info("✅ /churn-trends endpoint hit!")
        
#         # 1. Fetch all data including last_login_date with proper column selection
#         query = text("""
#             SELECT 
#                 user_id,
#                 age,
#                 gender,
#                 state,
#                 total_orders,
#                 total_spent,
#                 avg_order_frequency,
#                 CAST(total_logins AS INTEGER) as total_logins,
#                 total_time_spent,
#                 avg_time_per_session,
#                 abandoned_cart_count,
#                 last_login_date,
#                 EXTRACT(DAY FROM (CURRENT_DATE - last_login_date)) AS recency_days
#             FROM user_full_dataset
#             WHERE last_login_date IS NOT NULL
#         """)
        
#         full_df = pd.read_sql(query, engine)

#         if full_df.empty:
#             logging.warning("⚠️ No data found in the database")
#             return jsonify({"error": "No data found in the database"}), 404

#         # 2. Convert all columns to appropriate numeric types
#         numeric_cols = ['age', 'total_orders', 'total_spent', 'avg_order_frequency',
#                        'total_logins', 'total_time_spent', 'avg_time_per_session',
#                        'abandoned_cart_count', 'recency_days']
        
#         for col in numeric_cols:
#             if col in full_df.columns:
#                 full_df[col] = pd.to_numeric(full_df[col], errors='coerce').fillna(0)

#         # 3. Feature engineering
#         full_df["frequency"] = full_df["total_orders"] / (full_df["recency_days"].replace(0, np.nan) / 30 + 1)
#         full_df["frequency"] = full_df["frequency"].fillna(0)

#         # 4. One-hot encode categorical variables to match trained features
#         # Gender encoding
#         if 'gender' in full_df.columns:
#             full_df['gender_Male'] = (full_df['gender'] == 'Male').astype(int)
#             full_df['gender_Other'] = (full_df['gender'] == 'Other').astype(int)
#             full_df['gender_male'] = (full_df['gender'] == 'male').astype(int)  # Handle lowercase

#         # State encoding
#         states = ['Delhi', 'Gujarat', 'Haryana', 'Karnataka', 'Kerala', 
#                  'Madhya Pradesh', 'Maharashtra', 'Manipur', 'Mizoram', 
#                  'Rajasthan', 'Tamil Nadu', 'Uttar Pradesh', 'West Bengal']
        
#         for state in states:
#             col_name = f"state_{state.replace(' ', '_')}"
#             full_df[col_name] = (full_df['state'] == state).astype(int)

#         # 5. Ensure all trained features are present
#         missing_features = set(trained_features) - set(full_df.columns)
#         for feature in missing_features:
#             full_df[feature] = 0
#             logging.info(f"Added missing feature: {feature}")

#         # 6. Create working dataframe with just the features
#         df = full_df[trained_features].copy()

#         # 7. Scale features
#         try:
#             X_scaled = scaler.transform(df)
#             logging.info(f"Scaled data shape: {X_scaled.shape}")
#         except Exception as e:
#             logging.error(f"❌ Error scaling features: {e}")
#             return jsonify({"error": f"Error scaling features: {str(e)}"}), 500

#         # 8. Predict churn probabilities
#         try:
#             proba = model.predict_proba(X_scaled)
#             logging.info(f"Predict_proba shape: {proba.shape}")

#             if proba.shape[1] == 1:
#                 predictions = proba[:, 0]
#             else:
#                 predictions = proba[:, 1]

#             full_df["churn_probability"] = predictions
#             logging.info("Predictions added successfully")
#         except Exception as e:
#             logging.error(f"❌ Prediction error: {e}")
#             return jsonify({"error": f"Prediction failed: {str(e)}"}), 500

#         # 9. Group by month
#         try:
#             full_df["month"] = pd.to_datetime(full_df["last_login_date"]).dt.to_period("M")
#             trends = full_df.groupby("month")["churn_probability"].mean().reset_index()
#             trends['month'] = trends['month'].astype(str)
            
#             logging.info(f"Trends data:\n{trends.head()}")
#             return jsonify(trends.to_dict(orient="records"))
#         except Exception as e:
#             logging.error(f"❌ Error grouping by month: {e}")
#             return jsonify({"error": f"Could not generate trends: {str(e)}"}), 500

#     except Exception as e:
#         logging.error(f"❌ Error in /churn-trends: {e}", exc_info=True)
#         return jsonify({"error": f"Internal server error: {str(e)}"}), 500
# # ✅ Endpoint: Get churn by state
# @app.route('/churn-by-state', methods=['GET'])
# def get_churn_by_state():
#     try:
#         logging.info("✅ /churn-by-state endpoint hit!")
        
#         # Fetch data from the database
#         query = text("""
#             SELECT *, 
#                    EXTRACT(DAY FROM (CURRENT_DATE - last_login_date)) AS recency_days
#             FROM user_full_dataset
#         """)
#         full_df = pd.read_sql(query, engine)

#         if full_df.empty:
#             logging.warning("⚠️ No data found in the database")
#             return jsonify({"error": "No data found in the database"}), 404

#         # Feature engineering
#         full_df["frequency"] = full_df["total_orders"] / (full_df["recency_days"].replace(0, np.nan) / 30 + 1)
#         full_df["frequency"] = full_df["frequency"].fillna(0)

#         # Ensure all trained features are present
#         for col in trained_features:
#             if col not in full_df.columns:
#                 full_df[col] = 0
#                 logging.info(f"Added missing column: {col}")

#         # Create working dataframe with just the features
#         df = full_df[trained_features].copy()

#         # Scale features
#         try:
#             X_scaled = scaler.transform(df)
#             logging.info(f"Scaled data shape: {X_scaled.shape}")
#         except Exception as e:
#             logging.error(f"❌ Error scaling features: {e}")
#             return jsonify({"error": "Error scaling features"}), 500

#         # Predict churn probabilities
#         try:
#             proba = model.predict_proba(X_scaled)
#             logging.info(f"Predict_proba shape: {proba.shape}")

#             if proba.shape[1] == 1:
#                 predictions = proba[:, 0]
#             else:
#                 predictions = proba[:, 1]

#             full_df["churn_probability"] = predictions
#             logging.info("Predictions added successfully")
#         except Exception as e:
#             logging.error(f"❌ Prediction error: {e}")
#             return jsonify({"error": "Prediction failed"}), 500

#         # Group by state
#         try:
#             churn_by_state = full_df.groupby("state")["churn_probability"].mean().reset_index()
#             logging.info(f"Churn by state data:\n{churn_by_state.head()}")
#             return jsonify(churn_by_state.to_dict(orient="records"))
#         except Exception as e:
#             logging.error(f"❌ Error grouping by state: {e}")
#             return jsonify({"error": "Could not group by state"}), 500

#     except Exception as e:
#         logging.error(f"❌ Error in /churn-by-state: {e}", exc_info=True)
#         return jsonify({"error": f"Internal server error: {str(e)}"}), 500

# # ✅ Endpoint: Get churn by gender
# @app.route('/churn-by-gender', methods=['GET'])
# def get_churn_by_gender():
#     try:
#         logging.info("✅ /churn-by-gender endpoint hit!")
        
#         # Fetch data from the database
#         query = text("""
#             SELECT *, 
#                    EXTRACT(DAY FROM (CURRENT_DATE - last_login_date)) AS recency_days
#             FROM user_full_dataset
#         """)
#         full_df = pd.read_sql(query, engine)

#         if full_df.empty:
#             logging.warning("⚠️ No data found in the database")
#             return jsonify({"error": "No data found in the database"}), 404

#         # Feature engineering
#         full_df["frequency"] = full_df["total_orders"] / (full_df["recency_days"].replace(0, np.nan) / 30 + 1)
#         full_df["frequency"] = full_df["frequency"].fillna(0)

#         # Ensure all trained features are present
#         for col in trained_features:
#             if col not in full_df.columns:
#                 full_df[col] = 0
#                 logging.info(f"Added missing column: {col}")

#         # Create working dataframe with just the features
#         df = full_df[trained_features].copy()

#         # Scale features
#         try:
#             X_scaled = scaler.transform(df)
#             logging.info(f"Scaled data shape: {X_scaled.shape}")
#         except Exception as e:
#             logging.error(f"❌ Error scaling features: {e}")
#             return jsonify({"error": "Error scaling features"}), 500

#         # Predict churn probabilities
#         try:
#             proba = model.predict_proba(X_scaled)
#             logging.info(f"Predict_proba shape: {proba.shape}")

#             if proba.shape[1] == 1:
#                 predictions = proba[:, 0]
#             else:
#                 predictions = proba[:, 1]

#             full_df["churn_probability"] = predictions
#             logging.info("Predictions added successfully")
#         except Exception as e:
#             logging.error(f"❌ Prediction error: {e}")
#             return jsonify({"error": "Prediction failed"}), 500

#         # Group by gender
#         try:
#             churn_by_gender = full_df.groupby("gender")["churn_probability"].mean().reset_index()
#             logging.info(f"Churn by gender data:\n{churn_by_gender.head()}")
#             return jsonify(churn_by_gender.to_dict(orient="records"))
#         except Exception as e:
#             logging.error(f"❌ Error grouping by gender: {e}")
#             return jsonify({"error": "Could not group by gender"}), 500

#     except Exception as e:
#         logging.error(f"❌ Error in /churn-by-gender: {e}", exc_info=True)
#         return jsonify({"error": f"Internal server error: {str(e)}"}), 500

# # ✅ Endpoint: Get churn by age group
# @app.route('/churn-by-age', methods=['GET'])
# def get_churn_by_age():
#     try:
#         logging.info("✅ /churn-by-age endpoint hit!")
        
#         # Fetch data from the database
#         query = text("""
#             SELECT *, 
#                    EXTRACT(DAY FROM (CURRENT_DATE - last_login_date)) AS recency_days
#             FROM user_full_dataset
#         """)
#         full_df = pd.read_sql(query, engine)

#         if full_df.empty:
#             logging.warning("⚠️ No data found in the database")
#             return jsonify({"error": "No data found in the database"}), 404

#         # Feature engineering
#         full_df["frequency"] = full_df["total_orders"] / (full_df["recency_days"].replace(0, np.nan) / 30 + 1)
#         full_df["frequency"] = full_df["frequency"].fillna(0)

#         # Ensure all trained features are present
#         for col in trained_features:
#             if col not in full_df.columns:
#                 full_df[col] = 0
#                 logging.info(f"Added missing column: {col}")

#         # Create working dataframe with just the features
#         df = full_df[trained_features].copy()

#         # Scale features
#         try:
#             X_scaled = scaler.transform(df)
#             logging.info(f"Scaled data shape: {X_scaled.shape}")
#         except Exception as e:
#             logging.error(f"❌ Error scaling features: {e}")
#             return jsonify({"error": "Error scaling features"}), 500

#         # Predict churn probabilities
#         try:
#             proba = model.predict_proba(X_scaled)
#             logging.info(f"Predict_proba shape: {proba.shape}")

#             if proba.shape[1] == 1:
#                 predictions = proba[:, 0]
#             else:
#                 predictions = proba[:, 1]

#             full_df["churn_probability"] = predictions
#             logging.info("Predictions added successfully")
#         except Exception as e:
#             logging.error(f"❌ Prediction error: {e}")
#             return jsonify({"error": "Prediction failed"}), 500

#         # Create age groups
#         try:
#             full_df["age_group"] = pd.cut(
#                 full_df["age"],
#                 bins=[0, 18, 25, 35, 45, 55, np.inf],
#                 labels=["0-18", "19-25", "26-35", "36-45", "46-55", "55+"],
#                 right=False
#             )
            
#             # Group by age group
#             churn_by_age = full_df.groupby("age_group")["churn_probability"].mean().reset_index()
#             logging.info(f"Churn by age data:\n{churn_by_age.head()}")
#             return jsonify(churn_by_age.to_dict(orient="records"))
#         except Exception as e:
#             logging.error(f"❌ Error creating age groups: {e}")
#             return jsonify({"error": "Could not create age groups"}), 500

#     except Exception as e:
#         logging.error(f"❌ Error in /churn-by-age: {e}", exc_info=True)
#         return jsonify({"error": f"Internal server error: {str(e)}"}), 500

# # ✅ Endpoint: Get high-risk customers
# @app.route('/high-risk-customers', methods=['GET'])
# def get_high_risk_customers():
#     try:
#         logging.info("✅ /high-risk-customers endpoint hit!")
        
#         # Fetch data from the database
#         query = text("""
#             SELECT *, 
#                    EXTRACT(DAY FROM (CURRENT_DATE - last_login_date)) AS recency_days
#             FROM user_full_dataset
#         """)
#         full_df = pd.read_sql(query, engine)

#         if full_df.empty:
#             logging.warning("⚠️ No data found in the database")
#             return jsonify({"error": "No data found in the database"}), 404

#         # Feature engineering
#         full_df["frequency"] = full_df["total_orders"] / (full_df["recency_days"].replace(0, np.nan) / 30 + 1)
#         full_df["frequency"] = full_df["frequency"].fillna(0)

#         # Ensure all trained features are present
#         for col in trained_features:
#             if col not in full_df.columns:
#                 full_df[col] = 0
#                 logging.info(f"Added missing column: {col}")

#         # Create working dataframe with just the features
#         df = full_df[trained_features].copy()

#         # Scale features
#         try:
#             X_scaled = scaler.transform(df)
#             logging.info(f"Scaled data shape: {X_scaled.shape}")
#         except Exception as e:
#             logging.error(f"❌ Error scaling features: {e}")
#             return jsonify({"error": "Error scaling features"}), 500

#         # Predict churn probabilities
#         try:
#             proba = model.predict_proba(X_scaled)
#             logging.info(f"Predict_proba shape: {proba.shape}")

#             if proba.shape[1] == 1:
#                 predictions = proba[:, 0]
#             else:
#                 predictions = proba[:, 1]

#             full_df["churn_probability"] = predictions
#             full_df["risk_category"] = full_df["churn_probability"].apply(categorize_risk)
#             logging.info("Predictions and risk categories added successfully")
#         except Exception as e:
#             logging.error(f"❌ Prediction error: {e}")
#             return jsonify({"error": "Prediction failed"}), 500

#         # Filter high-risk customers
#         high_risk = full_df[full_df["risk_category"] == "High Risk"]
        
#         # Select relevant columns to return
#         result_columns = ['user_id', 'name', 'user_email', 'state', 'gender', 'age', 
#                          'total_orders', 'recency_days', 'churn_probability', 'risk_category']
        
#         if not high_risk.empty:
#             result = high_risk[result_columns].to_dict(orient="records")
#             logging.info(f"Found {len(result)} high-risk customers")
#             return jsonify(result)
#         else:
#             logging.info("No high-risk customers found")
#             return jsonify([])

#     except Exception as e:
#         logging.error(f"❌ Error in /high-risk-customers: {e}", exc_info=True)
#         return jsonify({"error": f"Internal server error: {str(e)}"}), 500

# # ✅ Run Flask server
# if __name__ == "__main__":
#     app.run(debug=True, port=5000)

import os
import pickle
import logging
import numpy as np
import pandas as pd
from flask import Flask, jsonify, request, make_response
from flask_cors import CORS
from sqlalchemy import create_engine, text
from sklearn.preprocessing import StandardScaler
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
DB_URL = os.getenv("DATABASE_URL")

if not DB_URL:
    logging.error("❌ DATABASE_URL is missing! Check your .env file.")
    raise ValueError("Missing DATABASE_URL")

# Flask app setup
app = Flask(__name__)
CORS(app)
logging.basicConfig(level=logging.DEBUG)

# Load model, scaler, and features
BASE_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE_DIR, "model.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "scaler.pkl")
FEATURES_PATH = os.path.join(BASE_DIR, "trained_features.pkl")

# Ensure required files exist
for path in [MODEL_PATH, SCALER_PATH, FEATURES_PATH]:
    if not os.path.exists(path):
        logging.error(f"❌ Missing required file: {path}")
        raise FileNotFoundError(f"❌ Model, Scaler, or Features file is missing!")

# Load model, scaler, and features
try:
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    with open(SCALER_PATH, "rb") as f:
        scaler = pickle.load(f)
    with open(FEATURES_PATH, "rb") as f:
        trained_features = pickle.load(f)
    logging.info("✅ Model, scaler, and features loaded successfully!")
except Exception as e:
    logging.error(f"❌ Error loading model, scaler, or features: {e}")
    exit()

# Database connection
try:
    engine = create_engine(DB_URL)
    with engine.connect() as conn:
        result = conn.execute(text("SELECT 1")).fetchone()
        logging.info(f"✅ Database Test Query Result: {result[0]}")
    logging.info("✅ Connected to database!")
except Exception as e:
    logging.error(f"❌ Database connection failed: {e}")
    exit()

def categorize_risk(probability):
    if probability >= 0.7:
        return 'High Risk'
    elif 0.3 <= probability < 0.7:
        return 'Medium Risk'
    else:
        return 'Low Risk'

def get_customer_data_with_predictions():
    try:
        query = text("""
        SELECT 
            user_id,
            user_email, 
            state,
            gender,
            age,
            total_orders,
            total_spent,
            avg_order_frequency,
            total_logins,
            total_time_spent,
            avg_time_per_session,
            abandoned_cart_count,
            last_login_date,
            EXTRACT(DAY FROM (CURRENT_DATE - last_login_date)) AS recency_days
            FROM user_full_dataset
     """)
        
        full_df = pd.read_sql(query, engine)

        if full_df.empty:
            logging.warning("⚠️ No data found in the database")
            return None

        # Convert numeric columns
        numeric_cols = ['age', 'total_orders', 'total_spent', 'avg_order_frequency',
                      'total_logins', 'total_time_spent', 'avg_time_per_session',
                      'abandoned_cart_count', 'recency_days']
        
        for col in numeric_cols:
            if col in full_df.columns:
                full_df[col] = pd.to_numeric(full_df[col], errors='coerce').fillna(0)

        # Feature engineering
        full_df["frequency"] = full_df["total_orders"] / (full_df["recency_days"].replace(0, np.nan) / 30 + 1)
        full_df["frequency"] = full_df["frequency"].fillna(0)

        # One-hot encode categorical variables
        if 'gender' in full_df.columns:
            full_df['gender_Male'] = (full_df['gender'] == 'Male').astype(int)
            full_df['gender_Other'] = (full_df['gender'] == 'Other').astype(int)
            full_df['gender_male'] = (full_df['gender'] == 'male').astype(int)

        states = ['Delhi', 'Gujarat', 'Haryana', 'Karnataka', 'Kerala', 
                 'Madhya Pradesh', 'Maharashtra', 'Manipur', 'Mizoram', 
                 'Rajasthan', 'Tamil Nadu', 'Uttar Pradesh', 'West Bengal']
        
        for state in states:
            col_name = f"state_{state.replace(' ', '_')}"
            full_df[col_name] = (full_df['state'] == state).astype(int)

        # Ensure all trained features are present
        missing_features = set(trained_features) - set(full_df.columns)
        for feature in missing_features:
            full_df[feature] = 0

        # Create working dataframe with just the features
        df = full_df[trained_features].copy()

        # Scale features and predict
        X_scaled = scaler.transform(df)
        proba = model.predict_proba(X_scaled)
        
        if proba.shape[1] == 1:
            predictions = proba[:, 0]
        else:
            predictions = proba[:, 1]

        full_df["churn_probability"] = predictions
        full_df["risk_category"] = full_df["churn_probability"].apply(categorize_risk)
        
        return full_df

    except Exception as e:
        logging.error(f"❌ Error in get_customer_data_with_predictions: {e}")
        return None

@app.route('/predict-churn-batch', methods=['POST'])
def predict_churn_batch():
    try:
        data = request.get_json()
        if not data or 'customers' not in data:
            return jsonify({"error": "Invalid request. Provide a JSON payload with a 'customers' list."}), 400

        customers = data['customers']
        if not customers:
            return jsonify({"error": "No customers provided in the 'customers' list."}), 400

        try:
            df = pd.DataFrame(customers)
        except ValueError as e:
            return jsonify({"error": "Invalid customer data format."}), 400

        for col in trained_features:
            if col not in df.columns:
                df[col] = 0

        df = df[trained_features]
        X_scaled = scaler.transform(df)
        proba = model.predict_proba(X_scaled)
        
        predictions = proba[:, 1] if proba.shape[1] > 1 else proba[:, 0]

        return jsonify({"predictions": predictions.tolist()}), 200
    except Exception as e:
        logging.error(f"❌ Error in /predict-churn-batch: {e}")
        return jsonify({"error": f"Internal server error: {e}"}), 500

@app.route('/total-customers', methods=['GET'])
def get_total_customers():
    try:
        query = text("SELECT COUNT(*) as total FROM user_full_dataset")
        with engine.connect() as conn:
            result = conn.execute(query).fetchone()
            return jsonify({"total_customers": int(result[0])})
    except Exception as e:
        logging.error(f"❌ Error in /total-customers: {e}")
        return jsonify({"error": str(e)}), 500
@app.route('/churn-stats', methods=['GET'])
def get_churn_stats():
    try:
        full_df = get_customer_data_with_predictions()
        if full_df is None:
            return jsonify({"error": "No data available"}), 404

        stats = {
            "total_customers": int(len(full_df)),
            "avg_churn_prob": float(full_df["churn_probability"].mean()),
            "high_risk": int((full_df["risk_category"] == "High Risk").sum()),
            "medium_risk": int((full_df["risk_category"] == "Medium Risk").sum()),
            "low_risk": int((full_df["risk_category"] == "Low Risk").sum())
        }
        return jsonify(stats)
    except Exception as e:
        logging.error(f"❌ Error in /churn-stats: {e}")
        return jsonify({"error": str(e)}), 500
@app.route('/retention-rate', methods=['GET'])
def get_retention_rate():
    try:
        query = text("""
            SELECT 
                COUNT(*) as total,
                SUM(CASE WHEN EXTRACT(DAY FROM (CURRENT_DATE - last_login_date)) <= 30 THEN 1 ELSE 0 END) as active
            FROM user_full_dataset
        """)
        with engine.connect() as conn:
            result = conn.execute(query).fetchone()
            retention_rate = (result[1] / result[0]) * 100 if result[0] > 0 else 0
            return jsonify({
                "retention_rate": float(retention_rate),
                "active_customers": int(result[1]),
                "total_customers": int(result[0])
            })
    except Exception as e:
        logging.error(f"❌ Error in /retention-rate: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/churn-trends', methods=['GET'])
def get_churn_trends():
    try:
        full_df = get_customer_data_with_predictions()
        if full_df is None:
            return jsonify({"error": "No data available"}), 404

        full_df["month"] = pd.to_datetime(full_df["last_login_date"]).dt.to_period("M")
        trends = full_df.groupby("month")["churn_probability"].mean().reset_index()
        trends['month'] = trends['month'].astype(str)
        
        return jsonify(trends.to_dict(orient="records"))
    except Exception as e:
        logging.error(f"❌ Error in /churn-trends: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/churn-by-state', methods=['GET'])
def get_churn_by_state():
    try:
        full_df = get_customer_data_with_predictions()
        if full_df is None:
            return jsonify({"error": "No data available"}), 404

        churn_by_state = full_df.groupby("state")["churn_probability"].mean().reset_index()
        return jsonify(churn_by_state.to_dict(orient="records"))
    except Exception as e:
        logging.error(f"❌ Error in /churn-by-state: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/churn-by-gender', methods=['GET'])
def get_churn_by_gender():
    try:
        full_df = get_customer_data_with_predictions()
        if full_df is None:
            return jsonify({"error": "No data available"}), 404

        churn_by_gender = full_df.groupby("gender")["churn_probability"].mean().reset_index()
        return jsonify(churn_by_gender.to_dict(orient="records"))
    except Exception as e:
        logging.error(f"❌ Error in /churn-by-gender: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/churn-by-age', methods=['GET'])
def get_churn_by_age():
    try:
        full_df = get_customer_data_with_predictions()
        if full_df is None:
            return jsonify({"error": "No data available"}), 404

        full_df["age_group"] = pd.cut(
            full_df["age"],
            bins=[0, 18, 25, 35, 45, 55, np.inf],
            labels=["0-18", "19-25", "26-35", "36-45", "46-55", "55+"],
            right=False
        )
        
        churn_by_age = full_df.groupby("age_group", observed=True)["churn_probability"].mean().reset_index()
        return jsonify(churn_by_age.to_dict(orient="records"))
    except Exception as e:
        logging.error(f"❌ Error in /churn-by-age: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/high-risk-customers', methods=['GET'])
def get_high_risk_customers():
    try:
        full_df = get_customer_data_with_predictions()
        if full_df is None:
            return jsonify({"error": "No data available"}), 404

        high_risk = full_df[full_df["risk_category"] == "High Risk"]
        result_columns = ['user_id', 'user_email', 'state', 'gender', 'age', 
                         'total_orders', 'recency_days', 'churn_probability', 'risk_category']
        
        return jsonify(high_risk[result_columns].to_dict(orient="records") if not high_risk.empty else [])
    except Exception as e:
        logging.error(f"❌ Error in /high-risk-customers: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/customer-segments', methods=['GET'])
def get_customer_segments():
    try:
        full_df = get_customer_data_with_predictions()
        if full_df is None:
            return jsonify({"error": "No data available"}), 404

        full_df["segment"] = "Low Value"
        full_df.loc[
            (full_df["recency_days"] <= 30) & 
            (full_df["total_orders"] >= 5) & 
            (full_df["total_spent"] >= 1000),
            "segment"
        ] = "High Value"
        
        full_df.loc[
            (full_df["recency_days"] <= 30) & 
            ((full_df["total_orders"] < 5) | (full_df["total_spent"] < 1000)),
            "segment"
        ] = "Potential"
        
        full_df.loc[
            (full_df["recency_days"] > 90),
            "segment"
        ] = "At Risk"

        segments = full_df.groupby("segment").agg({
            "user_id": "count",
            "churn_probability": "mean"
        }).reset_index()
        
        return jsonify(segments.to_dict(orient="records"))
    except Exception as e:
        logging.error(f"❌ Error in /customer-segments: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/export-data', methods=['GET'])
def export_data():
    try:
        full_df = get_customer_data_with_predictions()
        if full_df is None:
            return jsonify({"error": "No data available"}), 404

        format = request.args.get('format', 'json')
        
        if format == 'csv':
            from io import StringIO
            output = StringIO()
            full_df.to_csv(output, index=False)
            output.seek(0)
            
            response = make_response(output.getvalue())
            response.headers['Content-Type'] = 'text/csv'
            response.headers['Content-Disposition'] = 'attachment; filename=customers.csv'
            return response
        return jsonify(full_df.to_dict(orient="records"))
    except Exception as e:
        logging.error(f"❌ Error in /export-data: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/customer-details/<int:user_id>', methods=['GET'])
def get_customer_details(user_id):
    try:
        query = text("SELECT * FROM user_full_dataset WHERE user_id = :user_id")
        with engine.connect() as conn:
            result = conn.execute(query, {"user_id": user_id}).fetchone()
            if not result:
                return jsonify({"error": "Customer not found"}), 404
            return jsonify(dict(result))
    except Exception as e:
        logging.error(f"❌ Error in /customer-details: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/churned-customers', methods=['GET'])
def get_churned_customers():
    try:
        full_df = get_customer_data_with_predictions()
        if full_df is None:
            return jsonify({"error": "No data available"}), 404

        churned = full_df[
            (full_df["churn_probability"] >= 0.7) & 
            (full_df["recency_days"] > 90)
        ]
        
        result_columns = ['user_id', 'user_email', 'state', 'gender', 'age', 
                         'total_orders', 'recency_days', 'churn_probability']
        
        return jsonify(churned[result_columns].to_dict(orient="records") if not churned.empty else [])
    except Exception as e:
        logging.error(f"❌ Error in /churned-customers: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True, port=5000)