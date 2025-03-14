# import pandas as pd
# import pickle
# from sklearn.preprocessing import StandardScaler

# def save_scaler():
#     # Load dataset (Update with your actual dataset path)
#     df = pd.read_csv("your_dataset.csv")

#     # Print available columns (for debugging)
#     print("Available columns:", df.columns)

#     # Automatically select only numeric features (ignore categorical columns)
#     X = df.select_dtypes(include=['number'])  

#     # Initialize and fit scaler
#     scaler = StandardScaler()
#     scaler.fit(X)

#     # Save scaler
#     with open("scaler.pkl", "wb") as f:
#         pickle.dump(scaler, f)

#     print("✅ Scaler trained and saved successfully as `scaler.pkl`")

# # Run the function
# save_scaler()
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler

def save_scaler():
    """Trains and saves a StandardScaler for customer churn prediction."""
    try:
        # ✅ Load dataset (Update this path if needed)
        df = pd.read_csv("your_dataset.csv")

        # ✅ Print available columns (Debugging)
        print("Available columns:", df.columns)

        # ✅ Automatically select numeric features only
        X = df.select_dtypes(include=['number'])

        # ✅ Handle missing values (if any)
        X.fillna(0, inplace=True)

        # ✅ Train StandardScaler
        scaler = StandardScaler()
        scaler.fit(X)

        # ✅ Save scaler
        with open("scaler.pkl", "wb") as f:
            pickle.dump(scaler, f)

        print("✅ Scaler trained and saved successfully as `scaler.pkl`")

    except Exception as e:
        print(f"❌ Error saving scaler: {e}")

# ✅ Run function
if __name__ == "__main__":
    save_scaler()
