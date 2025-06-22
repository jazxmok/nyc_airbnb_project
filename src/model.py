# src/model.py

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import joblib
import os

def train_price_model(data_path: str = "data/NYC_2019.csv",
                      model_path: str = "src/price_model.pkl",
                      columns_path: str = "src/model_columns.pkl"):
    """Train a price prediction model and save both model + columns list."""

    # Load data
    df = pd.read_csv(data_path)

    # Drop rows with missing price
    df = df.dropna(subset=["price"])

    # One-hot encode necessary columns
    df = pd.get_dummies(df, columns=["neighbourhood_group", "room_type"], drop_first=False)

    # Define target and features
    y = df["price"]
    X = df.drop("price", axis=1)

    # Save list of final columns for prediction use
    model_columns = X.columns.tolist()

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    print(f"✅ Model trained. RMSE: ${rmse:.2f}")

    # Save model and columns
    joblib.dump(model, model_path)
    joblib.dump(model_columns, columns_path)
    print(f"✅ Model saved to {model_path}")
    print(f"✅ Columns saved to {columns_path}")

if __name__ == "__main__":
    train_price_model()
