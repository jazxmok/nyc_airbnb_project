# src/data_cleaning.py
import pandas as pd

def clean_airbnb_data(filepath: str) -> pd.DataFrame:
    """Load the raw NYC Airbnb CSV, perform cleaning steps, and return a DataFrame."""

    df = pd.read_csv(filepath)

    # 1️⃣ Drop columns not needed for modelling / dashboard
    cols_to_drop = ['id', 'name', 'host_id', 'host_name', 'last_review']
    df.drop(cols_to_drop, axis=1, inplace=True, errors='ignore')

    # 2️⃣ Handle missing values – reviews_per_month → 0 where NaN
    df['reviews_per_month'] = df['reviews_per_month'].fillna(0)

    # 3️⃣ Remove unreasonable prices (keep 1 < price < 1000 USD)
    df = df[(df['price'] > 0) & (df['price'] < 1000)]

    # 4️⃣ Filter out listings with unusually long minimum nights (> 30)
    df = df[df['minimum_nights'] <= 30]

    # 5️⃣ Reset index for cleanliness
    return df.reset_index(drop=True)

if __name__ == "__main__":
    cleaned = clean_airbnb_data("../data/nyc_airbnb.csv")
    cleaned.to_csv("../data/cleaned_airbnb.csv", index=False)
    print("✅ Cleaned dataset saved to data/cleaned_airbnb.csv")

# src/model.py
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import joblib

def train_price_model(data_path: str):
    df = pd.read_csv(data_path)

    # One‑hot encode categorical vars
    df = pd.get_dummies(df, columns=["neighbourhood_group", "room_type"], drop_first=True)

    X = df.drop("price", axis=1)
    y = df["price"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    rmse = mean_squared_error(y_test, preds, squared=False)
    print(f"Model RMSE: ${rmse:.2f}")

    joblib.dump(model, "../src/price_model.pkl")
    print("✅ Model saved as src/price_model.pkl")

if __name__ == "__main__":
    train_price_model("../data/cleaned_airbnb.csv")

# tests/test_model.py
import joblib
import pandas as pd

def test_model_prediction():
    model = joblib.load("../src/price_model.pkl")

    sample_input = {
        "minimum_nights": [3],
        "number_of_reviews": [10],
        "reviews_per_month": [1.2],
        "calculated_host_listings_count": [2],
        "availability_365": [200],
        "neighbourhood_group_Brooklyn": [1],
        "neighbourhood_group_Manhattan": [0],
        "neighbourhood_group_Queens": [0],
        "neighbourhood_group_Staten Island": [0],
        "room_type_Private room": [1],
        "room_type_Shared room": [0]
    }

    df = pd.DataFrame(sample_input)
    pred = model.predict(df)[0]
    print(f"Sample predicted price: ${pred:.2f}")

if __name__ == "__main__":
    test_model_prediction()

# dashboard.py
import streamlit as st
import pandas as pd
import joblib
import plotly.express as px

# Load cleaned data and model
DATA_PATH = "data/cleaned_airbnb.csv"
MODEL_PATH = "src/price_model.pkl"

df = pd.read_csv(DATA_PATH)
model = joblib.load(MODEL_PATH)

# Sidebar Filters
st.sidebar.header("🔍 Filter Listings")
neigh = st.sidebar.selectbox("Neighbourhood Group", df["neighbourhood_group"].unique())
room = st.sidebar.selectbox("Room Type", df["room_type"].unique())

filtered = df[(df["neighbourhood_group"] == neigh) & (df["room_type"] == room)]

# Main Title
st.title("🏙️ NYC Airbnb Explorer & Price Predictor")
st.markdown(f"Displaying **{len(filtered)}** listings for **{neigh}** – **{room}**")

# Summary Metrics
c1, c2, c3 = st.columns(3)
c1.metric("Avg Price", f"${filtered['price'].mean():.2f}")
c2.metric("Avg Reviews/Month", f"{filtered['reviews_per_month'].mean():.2f}")
c3.metric("Avg Availability", f"{filtered['availability_365'].mean():.0f} days")

# Price Distribution Plot
fig = px.histogram(filtered, x="price", nbins=30, title="Price Distribution")
st.plotly_chart(fig, use_container_width=True)

# --- Prediction Section ---
st.header("💡 Predict Listing Price")

min_nights = st.number_input("Minimum Nights", 1, 30, 3)
num_reviews = st.number_input("Number of Reviews", 0, 1000, 10)
rev_month = st.number_input("Reviews per Month", 0.0, 30.0, 1.2)
host_listings = st.number_input("Host Listings Count", 1, 50, 2)
avail_days = st.number_input("Availability (days/year)", 0, 365, 200)

# Prepare input for model
input_dict = {
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
}

input_df = pd.DataFrame(input_dict)

if st.button("Predict Price"):
    pred_price = model.predict(input_df)[0]
    st.success(f"Estimated Price: ${pred_price:.2f} per night")
