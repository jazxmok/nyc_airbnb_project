# src/model.py

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import joblib
import os

def train_price_model():
    data_path = "data/NYC_2019.csv"
    model_path = "src/price_model.pkl"
    columns_path = "src/model_columns.pkl"

    df = pd.read_csv(data_path)
    df.dropna(subset=["price"], inplace=True)

    # One-hot encode
    df = pd.get_dummies(df, columns=["neighbourhood_group", "room_type"], drop_first=False)

    # Save column structure
    model_columns = df.drop("price", axis=1).columns.tolist()

    # Prepare model
    X = df[model_columns]
    y = df["price"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)

    rmse = mean_squared_error(y_test, model.predict(X_test), squared=False)
    print(f"✅ Trained model. RMSE: ${rmse:.2f}")

    joblib.dump(model, model_path)
    joblib.dump(model_columns, columns_path)
    print("✅ Model and columns saved.")

if __name__ == "__main__":
    train_price_model()
