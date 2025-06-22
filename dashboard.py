import streamlit as st
import plotly.express as px
import pandas as pd
import joblib
import os
import sys

# --- Data loading helpers ---
def load_cleaned_data():
    data_path = os.path.join("data", "NYC_2019.csv")
    return pd.read_csv(data_path)
    
def load_model():
    model_path = os.path.join("src", "price_model.pkl")
    return joblib.load(model_path)

def prepare_input_dict(min_nights, num_reviews, rev_month, host_listings, avail_days, neigh, room):
    data = {
        "minimum_nights": [min_nights],
        "number_of_reviews": [num_reviews],
        "reviews_per_month": [rev_month],
        "calculated_host_listings_count": [host_listings],
        "availability_365": [avail_days],
        "neighbourhood_group_Brooklyn": [0],
        "neighbourhood_group_Manhattan": [0],
        "neighbourhood_group_Queens": [0],
        "neighbourhood_group_Staten Island": [0],
        "room_type_Private room": [0],
        "room_type_Shared room": [0]
    }

    if f"neighbourhood_group_{neigh}" in data:
        data[f"neighbourhood_group_{neigh}"] = [1]

    if f"room_type_{room}" in data:
        data[f"room_type_{room}"] = [1]

    return pd.DataFrame(data)


df = load_cleaned_data()
model = load_model()

# --- Sidebar filters ---
st.sidebar.title("🔍 Filter Listings")
neigh = st.sidebar.selectbox("Neighbourhood Group", df['neighbourhood_group'].unique())
room = st.sidebar.selectbox("Room Type", df['room_type'].unique())

filtered_df = df[(df['neighbourhood_group'] == neigh) & (df['room_type'] == room)]

# --- Title and summary ---
st.title("🏙️ NYC Airbnb Explorer & Price Predictor")
st.markdown(f"Showing **{len(filtered_df)} listings** for **{neigh} - {room}**")

col1, col2, col3 = st.columns(3)
col1.metric("Avg Price", f"${filtered_df['price'].mean():.2f}")
col2.metric("Avg Reviews/Month", f"{filtered_df['reviews_per_month'].mean():.2f}")
col3.metric("Avg Availability", f"{filtered_df['availability_365'].mean():.0f} days")

# --- Visualization ---
st.subheader("Price Distribution")
fig = px.histogram(filtered_df, x="price", nbins=30, title="Distribution of Prices")
st.plotly_chart(fig, use_container_width=True)

# --- Prediction Section ---
st.header("Predict Airbnb Listing Price")
st.markdown("Fill in the details below to estimate the price per night:")

min_nights = st.number_input("Minimum Nights", min_value=1, max_value=30, value=3)
num_reviews = st.number_input("Number of Reviews", min_value=0, max_value=1000, value=10)
rev_month = st.number_input("Reviews per Month", min_value=0.0, max_value=30.0, value=1.2)
host_listings = st.number_input("Host Listings Count", min_value=1, max_value=100, value=2)
avail_days = st.number_input("Availability per Year", min_value=0, max_value=365, value=200)

if st.button("Predict Price"):
    input_df = prepare_input_dict(min_nights, num_reviews, rev_month, host_listings, avail_days, neigh, room)
    pred_price = model.predict(input_df)[0]
    st.success(f"Estimated Price: **${pred_price:.2f}** per night")
