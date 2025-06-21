# src/model.py

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import joblib

def train_price_model(data_path: str, model_output_path: str):
    df = pd.read_csv(data_path)

    # One-hot encode categorical columns
    df = pd.get_dummies(df, columns=["neighbourhood_group", "room_type"], drop_first=True)

    X = df.drop("price", axis=1)
    y = df["price"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    rmse = mean_squared_error(y_test, preds, squared=False)
    print(f"✅ Model trained. RMSE: ${rmse:.2f}")

    joblib.dump(model, model_output_path)
    print(f"✅ Model saved to {model_output_path}")

if __name__ == "__main__":
    train_price_model("data/cleaned_airbnb.csv", "src/price_model.pkl")
