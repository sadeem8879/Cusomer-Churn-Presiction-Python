import os
import pandas as pd
from sqlalchemy import create_engine
from dotenv import load_dotenv

# ✅ Load environment variables
load_dotenv()
DATABASE_URL = os.getenv("DATABASE_URL")

# ✅ Database Connection
engine = create_engine(DATABASE_URL)

def fetch_data():
    """Fetches customer data from PostgreSQL."""
    query = """
        SELECT user_id, total_orders, total_spent, avg_order_frequency,
                total_logins, total_time_spent, avg_time_per_session,
                abandoned_cart_count, recency_days, age, gender, state
        FROM user_full_dataset;
    """
    df = pd.read_sql(query, engine)
    return df

if __name__ == "__main__":
    df = fetch_data()
    print("✅ Data fetched successfully!")
    print(df.head())