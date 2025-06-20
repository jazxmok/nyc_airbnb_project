# src/utils.py
import pandas as pd
import joblib
import os

MODEL_PATH = "src/price_model.pkl"
DATA_PATH = "data/cleaned_airbnb.csv"

# 🔹 Load cleaned data
def load_cleaned_data(filepath=DATA_PATH) -> pd.DataFrame:
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset not found: {filepath}")
    return pd.read_csv(filepath)

# 🔹 Load trained model
def load_model(filepath=MODEL_PATH):
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Model file not found: {filepath}")
    return joblib.load(filepath)

# 🔹 Get feature template for model input (important for prediction)
def get_model_feature_template() -> list:
    return [
        'minimum_nights',
        'number_of_reviews',
        'reviews_per_month',
        'calculated_host_listings_count',
        'availability_365',
        'neighbourhood_group_Brooklyn',
        'neighbourhood_group_Manhattan',
        'neighbourhood_group_Queens',
        'neighbourhood_group_Staten Island',
        'room_type_Private room',
        'room_type_Shared room'
    ]

# 🔹 Prepare user input into model-ready DataFrame
def prepare_input_dict(min_nights, num_reviews, rev_per_month, host_count, avail, neigh, room) -> pd.DataFrame:
    input_data = {
        "minimum_nights": [min_nights],
        "number_of_reviews": [num_reviews],
        "reviews_per_month": [rev_per_month],
        "calculated_host_listings_count": [host_count],
        "availability_365": [avail],
        "neighbourhood_group_Brooklyn": [1 if neigh == "Brooklyn" else 0],
        "neighbourhood_group_Manhattan": [1 if neigh == "Manhattan" else 0],
        "neighbourhood_group_Queens": [1 if neigh == "Queens" else 0],
        "neighbourhood_group_Staten Island": [1 if neigh == "Staten Island" else 0],
        "room_type_Private room": [1 if room == "Private room" else 0],
        "room_type_Shared room": [1 if room == "Shared room" else 0]
    }
    return pd.DataFrame(input_data)

# 🔹 Log function result for debugging
def log(msg: str):
    print(f"[INFO] {msg}")
