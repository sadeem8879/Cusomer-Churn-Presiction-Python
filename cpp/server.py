# import os
# import pickle
# import pandas as pd
# import numpy as np
# from flask import Flask, jsonify
# from flask_cors import CORS
# import logging
# from sqlalchemy import create_engine

# # ✅ Flask App Setup
# app = Flask(__name__)
# CORS(app)

# # ✅ Enable Logging
# logging.basicConfig(level=logging.DEBUG)

# # ✅ Database Connection
# DB_URL = os.getenv("DATABASE_URL")  # Make sure this is set in the .env file

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

# # ✅ API: Predict Churn by Gender, State, Age (For Admin Dashboard)
# @app.route("/predict-churn-stats", methods=["GET"])
# def predict_churn_statistics():
#     try:
#         # ✅ Connect to Database
#         engine = create_engine(DB_URL)

#         # ✅ Fetch User Data (Exclude churn column)
#         query = """
#         SELECT user_id, gender, state, age, recency_days, total_orders 
#         FROM user_full_dataset;
#         """  # Only relevant columns
        
#         df = pd.read_sql(query, engine)

#         if df.empty:
#             return jsonify({
#                 "churn_by_gender": [],
#                 "churn_by_state": [],
#                 "churn_by_age": []
#             })

#         # ✅ Feature Engineering (Ensure it matches ML training)
#         df["frequency"] = df["total_orders"] / ((df["recency_days"].replace(0, np.nan) / 30) + 1)
#         df["frequency"] = df["frequency"].fillna(0)

#         drop_columns = ["user_id"]  # Drop non-informative columns
#         df.drop(columns=[col for col in drop_columns if col in df.columns], errors="ignore", inplace=True)

#         # ✅ One-Hot Encoding for Categorical Features
#         categorical_columns = ["gender", "state"]
#         df = pd.get_dummies(df, columns=categorical_columns, drop_first=True)

#         # ✅ Align with Trained Model Features
#         for col in trained_features:
#             if col not in df.columns:
#                 df[col] = 0  # Add missing columns with default value 0

#         df = df[trained_features]  # Ensure correct column order

#         # ✅ Scale Features
#         X_scaled = scaler.transform(df)

#         # ✅ Predict Churn
#         df["churn"] = model.predict(X_scaled)

#         # ✅ Aggregate Predictions by Gender
#         churn_by_gender = df.groupby("gender_{}".format(df["gender"].iloc[0]))["churn"].sum().reset_index()
#         churn_by_gender.rename(columns={"churn": "customer_count"}, inplace=True)

#         # ✅ Aggregate Predictions by State
#         churn_by_state = df.groupby("state_{}".format(df["state"].iloc[0]))["churn"].sum().reset_index()
#         churn_by_state.rename(columns={"churn": "customer_count"}, inplace=True)

#         # ✅ Create Age Groups
#         df["age_group"] = pd.cut(df["age"], bins=[0, 17, 24, 34, 44, 54, np.inf], 
#                                  labels=["Under 18", "18-24", "25-34", "35-44", "45-54", "55+"], right=False)

#         # ✅ Aggregate Predictions by Age Group
#         churn_by_age = df.groupby("age_group")["churn"].sum().reset_index()
#         churn_by_age.rename(columns={"churn": "customer_count"}, inplace=True)

#         # ✅ Calculate Churn Percentage by Age
#         total_churned_customers = df["churn"].sum()
#         churn_by_age["churn_percentage"] = ((churn_by_age["customer_count"] / total_churned_customers) * 100).round(2)

#         return jsonify({
#             "churn_by_gender": churn_by_gender.to_dict(orient="records"),
#             "churn_by_state": churn_by_state.to_dict(orient="records"),
#             "churn_by_age": churn_by_age.to_dict(orient="records")
#         })

#     except Exception as e:
#         logging.error(f"❌ Error predicting churn statistics: {e}")
#         return jsonify({"error": str(e)}), 500

# # ✅ Run Flask Server
# if __name__ == "__main__":
#     app.run(debug=True)

import os
import pickle
import pandas as pd
import numpy as np
from flask import Flask, jsonify
from flask_cors import CORS
import logging
from sqlalchemy import create_engine

# ✅ Flask App Setup
app = Flask(__name__)
CORS(app)

# ✅ Enable Logging
logging.basicConfig(level=logging.DEBUG)

# ✅ Database Connection
DB_URL = os.getenv("DATABASE_URL")  # Make sure this is set in the .env file

# ✅ Load Model & Preprocessing Files
BASE_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE_DIR, "model.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "scaler.pkl")
FEATURES_PATH = os.path.join(BASE_DIR, "trained_features.pkl")

# ✅ Ensure Required Files Exist
for path in [MODEL_PATH, SCALER_PATH, FEATURES_PATH]:
    if not os.path.exists(path):
        logging.error(f"❌ Missing required file: {path}")
        raise FileNotFoundError(f"❌ Model, Scaler, or Features file is missing!")

# ✅ Load Model, Scaler & Features
with open(MODEL_PATH, "rb") as model_file:
    model = pickle.load(model_file)

with open(SCALER_PATH, "rb") as scaler_file:
    scaler = pickle.load(scaler_file)

with open(FEATURES_PATH, "rb") as features_file:
    trained_features = pickle.load(features_file)

# ✅ API: Predict Churn by Gender, State, Age (For Admin Dashboard)
@app.route("/predict-churn-stats", methods=["GET"])
def predict_churn_statistics():
    try:
        # ✅ Connect to Database
        engine = create_engine(DB_URL)

        # ✅ Fetch User Data (Exclude churn column)
        query = """
        SELECT user_id, gender, state, age, recency_days, total_orders 
        FROM user_full_dataset;
        """  # Only relevant columns
        
        df = pd.read_sql(query, engine)

        if df.empty:
            return jsonify({
                "churn_by_gender": [],
                "churn_by_state": [],
                "churn_by_age": []
            })

        # ✅ Feature Engineering (Ensure it matches ML training)
        df["frequency"] = df["total_orders"] / ((df["recency_days"].replace(0, np.nan) / 30) + 1)
        df["frequency"] = df["frequency"].fillna(0)

        drop_columns = ["user_id"]  # Drop non-informative columns
        df.drop(columns=[col for col in drop_columns if col in df.columns], errors="ignore", inplace=True)

        # ✅ One-Hot Encoding for Categorical Features
        categorical_columns = ["gender", "state"]
        df = pd.get_dummies(df, columns=categorical_columns, drop_first=True)

        # ✅ Align with Trained Model Features
        for col in trained_features:
            if col not in df.columns:
                df[col] = 0  # Add missing columns with default value 0

        df = df[trained_features]  # Ensure correct column order

        # ✅ Scale Features
        X_scaled = scaler.transform(df)

        # ✅ Predict Churn
        df["churn"] = model.predict(X_scaled)

        # ✅ Aggregate Predictions by Gender
        churn_by_gender = df.groupby("gender_{}".format(df["gender"].iloc[0]))["churn"].sum().reset_index()
        churn_by_gender.rename(columns={"churn": "customer_count"}, inplace=True)

        # ✅ Aggregate Predictions by State
        churn_by_state = df.groupby("state_{}".format(df["state"].iloc[0]))["churn"].sum().reset_index()
        churn_by_state.rename(columns={"churn": "customer_count"}, inplace=True)

        # ✅ Create Age Groups
        df["age_group"] = pd.cut(df["age"], bins=[0, 17, 24, 34, 44, 54, np.inf], 
                                 labels=["Under 18", "18-24", "25-34", "35-44", "45-54", "55+"], right=False)

        # ✅ Aggregate Predictions by Age Group
        churn_by_age = df.groupby("age_group")["churn"].sum().reset_index()
        churn_by_age.rename(columns={"churn": "customer_count"}, inplace=True)

        # ✅ Calculate Churn Percentage by Age
        total_churned_customers = df["churn"].sum()
        churn_by_age["churn_percentage"] = ((churn_by_age["customer_count"] / total_churned_customers) * 100).round(2)

        return jsonify({
            "churn_by_gender": churn_by_gender.to_dict(orient="records"),
            "churn_by_state": churn_by_state.to_dict(orient="records"),
            "churn_by_age": churn_by_age.to_dict(orient="records")
        })

    except Exception as e:
        logging.error(f"❌ Error predicting churn statistics: {e}")
        return jsonify({"error": str(e)}), 500

# ✅ Run Flask Server
if __name__ == "__main__":
    app.run(debug=True)
