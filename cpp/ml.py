# import os
# import pickle
# import logging
# import numpy as np
# import pandas as pd
# from dotenv import load_dotenv
# from sqlalchemy import create_engine
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.impute import SimpleImputer
# from sklearn.metrics import accuracy_score

# # ✅ Configure Logging
# logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# # ✅ Load environment variables
# load_dotenv()
# DB_URL = os.getenv("DATABASE_URL")

# if not DB_URL:
#     logging.error("❌ DATABASE_URL not found! Please set it in the .env file.")
#     exit()

# # ✅ Connect to PostgreSQL and Fetch Data
# try:
#     engine = create_engine(DB_URL)
#     df = pd.read_sql("SELECT * FROM user_full_dataset;", engine)
#     logging.info("✅ Data fetched successfully!")
# except Exception as e:
#     logging.error(f"❌ Error fetching data: {e}")
#     exit()

# # ✅ Handle Missing Values
# numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
# df[numeric_columns] = SimpleImputer(strategy="mean").fit_transform(df[numeric_columns])

# # ✅ Convert `recency_days` to numeric
# if "recency_days" in df.columns:
#     df["recency_days"] = pd.to_numeric(df["recency_days"], errors="coerce")

# # ✅ Feature Engineering - Frequency Calculation
# if "total_orders" in df.columns and "recency_days" in df.columns:
#     df["frequency"] = df["total_orders"] / ((df["recency_days"].replace(0, np.nan) / 30) + 1)  
#     df["frequency"] = df["frequency"].fillna(0)

# # ✅ Define Churn Threshold
# CHURN_THRESHOLD = 90
# df["churn"] = np.where(df["recency_days"] > CHURN_THRESHOLD, 1, 0)

# # ✅ Drop Non-Informative Columns
# drop_columns = ["user_id", "user_email", "user_name", "last_order_date", "last_login_date"]
# df.drop(columns=[col for col in drop_columns if col in df.columns], errors="ignore", inplace=True)

# # ✅ Dynamically Convert Categorical Variables
# categorical_columns = df.select_dtypes(include=["object"]).columns.tolist()
# df = pd.get_dummies(df, columns=categorical_columns, drop_first=True)

# # ✅ Ensure All Columns Are Numeric
# df = df.apply(pd.to_numeric, errors='coerce')

# # ✅ Define Features and Target
# X = df.drop(columns=["churn"])
# y = df["churn"]

# # ✅ Save Feature Names for Consistency
# trained_features = X.columns.tolist()

# # ✅ Train-Test Split
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# # ✅ Scale Features
# scaler = StandardScaler()
# X_train_scaled = scaler.fit_transform(X_train)
# X_test_scaled = scaler.transform(X_test)

# # ✅ Train Model
# model = RandomForestClassifier(n_estimators=200, random_state=42)
# model.fit(X_train_scaled, y_train)

# # ✅ Evaluate Model
# accuracy = accuracy_score(y_test, model.predict(X_test_scaled))
# logging.info(f"✅ Model Accuracy: {accuracy:.4f}")

# # ✅ Save Model & Scaler
# BASE_DIR = os.path.dirname(__file__)

# with open(os.path.join(BASE_DIR, "model.pkl"), "wb") as f:
#     pickle.dump(model, f)

# with open(os.path.join(BASE_DIR, "scaler.pkl"), "wb") as f:
#     pickle.dump(scaler, f)

# with open(os.path.join(BASE_DIR, "trained_features.pkl"), "wb") as f:
#     pickle.dump(trained_features, f)

# logging.info("✅ Model, Scaler & Features saved successfully!")

import os
import pickle
import logging
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score

# ✅ Configure Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# ✅ Load environment variables
load_dotenv()
DB_URL = os.getenv("DATABASE_URL")

if not DB_URL:
    logging.error("❌ DATABASE_URL not found! Please set it in the .env file.")
    exit()

# ✅ Connect to PostgreSQL and Fetch Data
try:
    engine = create_engine(DB_URL)
    df = pd.read_sql("SELECT * FROM user_full_dataset;", engine)
    logging.info("✅ Data fetched successfully!")
except Exception as e:
    logging.error(f"❌ Error fetching data: {e}")
    exit()

# ✅ Handle Missing Values
numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
df[numeric_columns] = SimpleImputer(strategy="mean").fit_transform(df[numeric_columns])

# ✅ Convert `recency_days` to numeric
if "recency_days" in df.columns:
    df["recency_days"] = pd.to_numeric(df["recency_days"], errors="coerce")

# ✅ Feature Engineering - Frequency Calculation
if "total_orders" in df.columns and "recency_days" in df.columns:
    df["frequency"] = df["total_orders"] / ((df["recency_days"].replace(0, np.nan) / 30) + 1)  
    df["frequency"] = df["frequency"].fillna(0)

# ✅ Define Churn Threshold
CHURN_THRESHOLD = 90
df["churn"] = np.where(df["recency_days"] > CHURN_THRESHOLD, 1, 0)

# ✅ Drop Non-Informative Columns
drop_columns = ["user_id", "user_email", "user_name", "last_order_date", "last_login_date"]
df.drop(columns=[col for col in drop_columns if col in df.columns], errors="ignore", inplace=True)

# ✅ Dynamically Convert Categorical Variables
categorical_columns = df.select_dtypes(include=["object"]).columns.tolist()
df = pd.get_dummies(df, columns=categorical_columns, drop_first=True)

# ✅ Ensure All Columns Are Numeric
df = df.apply(pd.to_numeric, errors='coerce')

# ✅ Define Features and Target
X = df.drop(columns=["churn"])
y = df["churn"]

# ✅ Save Feature Names for Consistency
trained_features = X.columns.tolist()

# ✅ Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# ✅ Scale Features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ✅ Train Model
model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train_scaled, y_train)

# ✅ Evaluate Model
accuracy = accuracy_score(y_test, model.predict(X_test_scaled))
logging.info(f"✅ Model Accuracy: {accuracy:.4f}")

# ✅ Save Model & Scaler
BASE_DIR = os.path.dirname(__file__)

with open(os.path.join(BASE_DIR, "model.pkl"), "wb") as f:
    pickle.dump(model, f)

with open(os.path.join(BASE_DIR, "scaler.pkl"), "wb") as f:
    pickle.dump(scaler, f)

with open(os.path.join(BASE_DIR, "trained_features.pkl"), "wb") as f:
    pickle.dump(trained_features, f)

logging.info("✅ Model, Scaler & Features saved successfully!")
