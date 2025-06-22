# dashboard.py

import streamlit as st
import plotly.express as px
import pandas as pd
import os
import sys

# 1. Ensure we can import from src/ — works locally and on Streamlit Cloud
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from utils import load_cleaned_data, load_model, prepare_input_dict

# 2. Load dataset and pre-trained model
df = load_cleaned_data()
model = load_model()

# 3. Sidebar filters
st.sidebar.title("🔍 Filter Listings")
neigh = st.sidebar.selectbox("Neighbourhood Group", df["neighbourhood_group"].unique())
room = st.sidebar.selectbox("Room Type", df["room_type"].unique())

filtered_df = df[(df["neighbourhood_group"] == neigh) & (df["room_type"] == room)]

# 4. Main dashboard content
st.title("🏙️ NYC Airbnb Explorer & Price Predictor")
st.markdown(f"Showing **{len(filtered_df)}** listings in **{neigh} - {room}**")

col1, col2, col3 = st.columns(3)
col1.metric("Avg Price", f"${filtered_df['price'].mean():.2f}")
col2.metric("Avg Reviews/Month", f"{filtered_df['reviews_per_month'].mean():.2f}")
col3.metric("Avg Availability", f"{filtered_df['availability_365'].mean():.0f} days")

st.subheader("💸 Price Distribution")
fig = px.histogram(filtered_df, x="price", nbins=30)
st.plotly_chart(fig, use_container_width=True)

# 5. Prediction section
st.header("🧠 Predict Airbnb Listing Price")
st.markdown("Use the form below to estimate a listing's price.")

min_nights = st.number_input("Minimum Nights", 1, 30, 3)
num_reviews = st.number_input("Number of Reviews", 0, 1000, 10)
rev_month = st.number_input("Reviews per Month", 0.0, 30.0, 1.2)
host_listings = st.number_input("Host Listings Count", 1, 100, 2)
avail_days = st.number_input("Availability per Year", 0, 365, 200)

# 6. Generate prediction and display result
if st.button("🔮 Predict Price"):
    input_df = prepare_input_dict(neigh, room, min_nights, num_reviews, rev_month, host_listings, avail_days)
    pred_price = model.predict(input_df)[0]
    st.success(f"Estimated Price: **${pred_price:.2f}** per night")
