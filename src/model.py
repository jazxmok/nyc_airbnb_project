# src/model.py

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import joblib
import os

def train_price_model(data_path: str = "data/cleaned_airbnb.csv",
                      model_path: str = "src/price_model.pkl"):
    """Train a price prediction model and save it as a .pkl file."""

    # Load cleaned data
    df = pd.read_csv(data_path)

    # One-hot encode categorical features
    df = pd.get_dummies(df, columns=["neighbourhood_group", "room_type"], drop_first=True)

    # Split features and target
    X = df.drop("price", axis=1)
    y = df["price"]

    # Split into train/test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train linear regression model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Evaluate model
    y_pred = model.predict(X_test)
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    print(f"✅ Model trained. RMSE: ${rmse:.2f}")

    # Save model to file
    joblib.dump(model, model_path)
    print(f"✅ Model saved to {model_path}")

if __name__ == "__main__":
    train_price_model()

