# import os
# import pickle
# import logging
# import numpy as np
# import pandas as pd
# from dotenv import load_dotenv
# from sqlalchemy import create_engine
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler, OneHotEncoder
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.impute import SimpleImputer
# from sklearn.metrics import accuracy_score
# from sqlalchemy import text

# # ✅ Configure Logging
# logging.basicConfig(level=logging.INFO)

# # ✅ Load Environment Variables
# load_dotenv()
# DB_URL = os.getenv("DATABASE_URL")

# if not DB_URL:
#     logging.error("❌ DATABASE_URL missing! Check .env file.")
#     exit()
# # ✅ Establish Database Connection
# try:
#     engine = create_engine(DB_URL, pool_pre_ping=True)
#     with engine.connect() as conn:
#         result = conn.execute(text("SELECT 1")).fetchone()  # ✅ FIXED
#         logging.info(f"✅ Database Test Query Result: {result[0]}")

#     logging.info("✅ Connected to database!")
# except Exception as e:
#     logging.error(f"❌ Database connection failed: {e}")
#     exit()


# # ✅ Fetch Data
# try:
#     df = pd.read_sql("SELECT * FROM user_full_dataset;", engine)
#     logging.info("✅ Data fetched successfully!")
# except Exception as e:
#     logging.error(f"❌ Error fetching data: {e}")
#     exit()

# # ✅ Handle Missing Values
# numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
# df[numeric_columns] = SimpleImputer(strategy="mean").fit_transform(df[numeric_columns])

# # ✅ Feature Engineering
# df["frequency"] = df["total_orders"] / ((df["recency_days"].replace(0, np.nan) / 30) + 1)
# df["frequency"] = df["frequency"].fillna(0)

# # ✅ Dynamic Churn Threshold (75th percentile of inactivity)
# threshold = df["recency_days"].quantile(0.75)
# df["churn"] = np.where(df["recency_days"] > threshold, 1, 0)

# # ✅ Drop Unnecessary Columns
# df.drop(columns=["user_id", "user_email"], errors="ignore", inplace=True)

# # ✅ One-Hot Encoding for Categorical Variables
# categorical_cols = df.select_dtypes(include=["object"]).columns.tolist()
# encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)

# df_encoded = pd.DataFrame(encoder.fit_transform(df[categorical_cols]))
# df_encoded.columns = encoder.get_feature_names_out(categorical_cols)

# df = df.drop(columns=categorical_cols).reset_index(drop=True)
# df = pd.concat([df, df_encoded], axis=1)

# # ✅ Feature & Label Separation
# X, y = df.drop(columns=["churn"]), df["churn"]
# trained_features = X.columns.tolist()

# # ✅ Train-Test Split
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # ✅ Feature Scaling
# scaler = StandardScaler()
# X_train_scaled, X_test_scaled = scaler.fit_transform(X_train), scaler.transform(X_test)

# # ✅ Model Training
# model = RandomForestClassifier(n_estimators=200, random_state=42)
# model.fit(X_train_scaled, y_train)

# # ✅ Model Evaluation
# accuracy = accuracy_score(y_test, model.predict(X_test_scaled))
# logging.info(f"✅ Model Accuracy: {accuracy:.4f}")

# # ✅ Save Model, Scaler, and Features
# with open("model.pkl", "wb") as f:
#     pickle.dump(model, f)

# with open("scaler.pkl", "wb") as f:
#     pickle.dump(scaler, f)

# with open("trained_features.pkl", "wb") as f:
#     pickle.dump(trained_features, f)

# logging.info("✅ Model saved successfully!")


# import os
# import pickle
# import logging
# import numpy as np
# import pandas as pd
# from dotenv import load_dotenv
# from sqlalchemy import create_engine, text
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler, OneHotEncoder
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.impute import SimpleImputer
# from sklearn.metrics import accuracy_score

# # ✅ Configure Logging
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# # ✅ Load Environment Variables
# load_dotenv()
# DB_URL = os.getenv("DATABASE_URL")

# if not DB_URL:
#     logger.error("❌ DATABASE_URL missing! Check .env file.")
#     exit()

# # ✅ Establish Database Connection
# try:
#     engine = create_engine(DB_URL, pool_pre_ping=True)
#     with engine.connect() as conn:
#         result = conn.execute(text("SELECT 1")).fetchone()  # Test connection
#         logger.info(f"✅ Database Test Query Result: {result[0]}")
#     logger.info("✅ Connected to database!")
# except Exception as e:
#     logger.error(f"❌ Database connection failed: {e}")
#     exit()

# # ✅ Fetch Data
# try:
#     df = pd.read_sql("SELECT * FROM user_full_dataset;", engine)
#     logger.info("✅ Data fetched successfully!")
# except Exception as e:
#     logger.error(f"❌ Error fetching data: {e}")
#     exit()

# # ✅ Handle Missing Values
# numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
# df[numeric_columns] = SimpleImputer(strategy="mean").fit_transform(df[numeric_columns])

# # ✅ Feature Engineering
# df["frequency"] = df["total_orders"] / (df["recency_days"].replace(0, np.nan) / 30 + 1)
# df["frequency"] = df["frequency"].fillna(0)

# # ✅ Dynamic Churn Threshold (75th percentile of inactivity)
# threshold = df["recency_days"].quantile(0.75)
# df["churn"] = np.where(df["recency_days"] > threshold, 1, 0)

# # ✅ Drop Unnecessary Columns
# df.drop(columns=["user_id", "user_email"], errors="ignore", inplace=True)

# # ✅ One-Hot Encoding for Categorical Variables
# categorical_cols = df.select_dtypes(include=["object"]).columns.tolist()
# encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)

# df_encoded = pd.DataFrame(encoder.fit_transform(df[categorical_cols]))
# df_encoded.columns = encoder.get_feature_names_out(categorical_cols)

# df = df.drop(columns=categorical_cols).reset_index(drop=True)
# df = pd.concat([df, df_encoded], axis=1)

# # ✅ Feature & Label Separation
# X, y = df.drop(columns=["churn"]), df["churn"]
# trained_features = X.columns.tolist()

# # ✅ Train-Test Split
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # ✅ Feature Scaling
# scaler = StandardScaler()
# X_train_scaled, X_test_scaled = scaler.fit_transform(X_train), scaler.transform(X_test)

# # ✅ Model Training
# model = RandomForestClassifier(n_estimators=200, random_state=42)
# model.fit(X_train_scaled, y_train)

# # ✅ Model Evaluation
# accuracy = accuracy_score(y_test, model.predict(X_test_scaled))
# logger.info(f"✅ Model Accuracy: {accuracy:.4f}")

# # ✅ Save Model, Scaler, and Features
# with open("model.pkl", "wb") as f:
#     pickle.dump(model, f)

# with open("scaler.pkl", "wb") as f:
#     pickle.dump(scaler, f)

# with open("trained_features.pkl", "wb") as f:
#     pickle.dump(trained_features, f)

# logger.info("✅ Model saved successfully!")
import os
import pickle
import logging
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score

# ✅ Configure Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ✅ Load Environment Variables
load_dotenv()
DB_URL = os.getenv("DATABASE_URL")

if not DB_URL:
    logger.error("❌ DATABASE_URL missing! Check .env file.")
    exit()

# ✅ Establish Database Connection
try:
    engine = create_engine(DB_URL, pool_pre_ping=True) #Added pool_pre_ping
    with engine.connect() as conn:
        result = conn.execute(text("SELECT 1")).fetchone()  # Test connection
        logger.info(f"✅ Database Test Query Result: {result[0]}")
    logger.info("✅ Connected to database!")
except Exception as e:
    logger.error(f"❌ Database connection failed: {e}")
    exit()

# ✅ Fetch Data
try:
    df = pd.read_sql(text("SELECT * FROM user_full_dataset;"), engine) # changed from string to text
    logger.info("✅ Data fetched successfully!")
except Exception as e:
    logger.error(f"❌ Error fetching data: {e}")
    exit()

# ✅ Handle Missing Values
numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
df[numeric_columns] = SimpleImputer(strategy="mean").fit_transform(df[numeric_columns])

# ✅ Feature Engineering
df["frequency"] = df["total_orders"] / (df["recency_days"].replace(0, np.nan) / 30 + 1)
df["frequency"] = df["frequency"].fillna(0)

# ✅ Dynamic Churn Threshold (75th percentile of inactivity)
threshold = df["recency_days"].quantile(0.75)
df["churn"] = np.where(df["recency_days"] > threshold, 1, 0)

# ✅ Drop Unnecessary Columns
df.drop(columns=["user_id", "user_email"], errors="ignore", inplace=True)

# ✅ One-Hot Encoding for Categorical Variables
categorical_cols = df.select_dtypes(include=["object"]).columns.tolist()
encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)

df_encoded = pd.DataFrame(encoder.fit_transform(df[categorical_cols]))
df_encoded.columns = encoder.get_feature_names_out(categorical_cols)

df = df.drop(columns=categorical_cols).reset_index(drop=True)
df = pd.concat([df, df_encoded], axis=1)

# ✅ Feature & Label Separation
X, y = df.drop(columns=["churn"]), df["churn"]
trained_features = X.columns.tolist()

# ✅ Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ✅ Feature Scaling
scaler = StandardScaler()
X_train_scaled, X_test_scaled = scaler.fit_transform(X_train), scaler.transform(X_test)

# ✅ Model Training
model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train_scaled, y_train)

# ✅ Model Evaluation
accuracy = accuracy_score(y_test, model.predict(X_test_scaled))
logger.info(f"✅ Model Accuracy: {accuracy:.4f}")

# ✅ Save Model, Scaler, and Features
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

with open("trained_features.pkl", "wb") as f:
    pickle.dump(trained_features, f)

logger.info("✅ Model saved successfully!")
