# dashboard.py
import streamlit as st
import pandas as pd
import joblib
import plotly.express as px

# Load data and model
df = pd.read_csv("data/cleaned_airbnb.csv")
model = joblib.load("src/price_model.pkl")

# Sidebar Filters
st.sidebar.header("Filter Listings")
neighborhood = st.sidebar.selectbox("Neighbourhood Group", df['neighbourhood_group'].unique())
room_type = st.sidebar.selectbox("Room Type", df['room_type'].unique())

filtered_df = df[(df['neighbourhood_group'] == neighborhood) & (df['room_type'] == room_type)]

# Title and Summary
st.title("🏙️ NYC Airbnb Explorer & Price Predictor")
st.markdown(f"Showing **{len(filtered_df)}** listings for **{neighborhood}** with **{room_type}**")

# Summary Metrics
col1, col2, col3 = st.columns(3)
col1.metric("Avg Price", f"${filtered_df['price'].mean():.2f}")
col2.metric("Avg Reviews/Month", f"{filtered_df['reviews_per_month'].mean():.2f}")
col3.metric("Avg Availability", f"{filtered_df['availability_365'].mean():.0f} days")

# Visualization
fig = px.histogram(filtered_df, x="price", nbins=30, title="Price Distribution")
st.plotly_chart(fig)

# Prediction Section
st.header("💡 Predict Airbnb Price")
st.markdown("Enter values to estimate the nightly price:")

# User inputs
min_nights = st.number_input("Minimum Nights", 1, 30, 3)
reviews = st.number_input("Number of Reviews", 0, 1000, 10)
reviews_month = st.number_input("Reviews per Month", 0.0, 30.0, 1.0)
host_listings = st.number_input("Host Listings Count", 1, 50, 2)
availability = st.number_input("Availability (days/year)", 0, 365, 200)

# Encoding inputs
input_dict = {
    'minimum_nights': [min_nights],
    'number_of_reviews': [reviews],
    'reviews_per_month': [reviews_month],
    'calculated_host_listings_count': [host_listings],
    'availability_365': [availability],
    'neighbourhood_group_Brooklyn': [1 if neighborhood == "Brooklyn" else 0],
    'neighbourhood_group_Manhattan': [1 if neighborhood == "Manhattan" else 0],
    'neighbourhood_group_Queens': [1 if neighborhood == "Queens" else 0],
    'neighbourhood_group_Staten Island': [1 if neighborhood == "Staten Island" else 0],
    'room_type_Private room': [1 if room_type == "Private room" else 0],
    'room_type_Shared room': [1 if room_type == "Shared room" else 0]
}

input_df = pd.DataFrame(input_dict)

# Make prediction
if st.button("Predict Price"):
    price_pred = model.predict(input_df)[0]
    st.success(f"Estimated Price: ${price_pred:.2f} per night")
