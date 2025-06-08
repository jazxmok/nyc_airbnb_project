# tests/test_model.py
import joblib
import pandas as pd

def test_model_prediction():
    model = joblib.load("../src/price_model.pkl")
    
    sample = {
        'minimum_nights': [3],
        'number_of_reviews': [5],
        'reviews_per_month': [0.5],
        'calculated_host_listings_count': [2],
        'availability_365': [200],
        'neighbourhood_group_Brooklyn': [1],
        'neighbourhood_group_Manhattan': [0],
        'neighbourhood_group_Queens': [0],
        'neighbourhood_group_Staten Island': [0],
        'room_type_Private room': [1],
        'room_type_Shared room': [0]
    }

    sample_df = pd.DataFrame(sample)
    prediction = model.predict(sample_df)
    print(f"Test Prediction: ${prediction[0]:.2f}")

if __name__ == "__main__":
    test_model_prediction()
