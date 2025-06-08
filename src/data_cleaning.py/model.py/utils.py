# src/data_cleaning.py
import pandas as pd

def clean_airbnb_data(filepath):
    df = pd.read_csv(filepath)

    # Drop unused columns
    df.drop(['id', 'name', 'host_id', 'host_name', 'last_review'], axis=1, inplace=True)

    # Fill missing reviews_per_month with 0
    df['reviews_per_month'].fillna(0, inplace=True)

    # Remove outliers in price
    df = df[(df['price'] > 0) & (df['price'] < 1000)]

    # Filter minimum nights to reasonable values
    df = df[df['minimum_nights'] <= 30]

    return df

if __name__ == "__main__":
    cleaned_df = clean_airbnb_data("../data/nyc_airbnb.csv")
    cleaned_df.to_csv("../data/cleaned_airbnb.csv", index=False)
    print("Cleaned dataset saved to /data/cleaned_airbnb.csv")


# src/model.py
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import joblib

def train_price_model(data_path):
    df = pd.read_csv(data_path)

    # Convert categorical variables
    df = pd.get_dummies(df, columns=['neighbourhood_group', 'room_type'], drop_first=True)

    X = df.drop('price', axis=1)
    y = df['price']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    rmse = mean_squared_error(y_test, predictions, squared=False)
    print(f"Model RMSE: ${rmse:.2f}")

    # Save model
    joblib.dump(model, "../src/price_model.pkl")
    print("Model saved as price_model.pkl")

if __name__ == "__main__":
    train_price_model("../data/cleaned_airbnb.csv")
