# src/model.py

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import joblib
import os

def train_price_model():
    # Load data
    df = pd.read_csv("data/NYC_2019.csv")

    # One-hot encoding
    df = pd.get_dummies(df, columns=["neighbourhood_group", "room_type"], drop_first=True)

    X = df.drop("price", axis=1)
    y = df["price"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)

    rmse = mean_squared_error(y_test, model.predict(X_test), squared=False)
    print(f"✅ Model trained. RMSE: ${rmse:.2f}")

    # Save model and column structure
    joblib.dump(model, "src/price_model.pkl")
    joblib.dump(X.columns.tolist(), "src/model_columns.pkl")
    print("✅ Model and columns saved")

if __name__ == "__main__":
    train_price_model()
