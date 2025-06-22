# src/utils.py

import pandas as pd
import joblib
import os

def load_cleaned_data():
    data_path = os.path.join("data", "nyc_2019.csv")
    return pd.read_csv(data_path)

def load_model():
    model_path = os.path.join("model", "model.pkl")
    return joblib.load(model_path)

def prepare_input_dict(min_nights, num_reviews, rev_month, host_listings, avail_days, neigh, room):
    return pd.DataFrame({
        "minimum_nights": [min_nights],
        "number_of_reviews": [num_reviews],
        "reviews_per_month": [rev_month],
        "calculated_host_listings_count": [host_listings],
        "availability_365": [avail_days],
        "neighbourhood_group_Brooklyn": [1 if neigh == "Brooklyn" else 0],
        "neighbourhood_group_Manhattan": [1 if neigh == "Manhattan" else 0],
        "neighbourhood_group_Queens": [1 if neigh == "Queens" else 0],
        "neighbourhood_group_Staten Island": [1 if neigh == "Staten Island" else 0],
        "room_type_Private room": [1 if room == "Private room" else 0],
        "room_type_Shared room": [1 if room == "Shared room" else 0]
    })
