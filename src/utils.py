# utils.py
import pandas as pd
import joblib
import os

def load_cleaned_data(data_path):
    """Load the cleaned Airbnb data"""
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found at: {data_path}")
    return pd.read_csv(data_path)

def load_model():
    """Load the trained model"""
    model_path = os.path.join(os.path.dirname(__file__), "models", "random_forest_model.joblib")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at: {model_path}")
    return joblib.load(model_path)
