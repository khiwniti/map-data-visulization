import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

def preprocess_data(data):
    # Example preprocessing steps
    data = data.dropna(subset=['lat', 'lng'])
    data['price_range'] = data['price_range'].astype('category').cat.codes
    data['category'] = data['category'].astype('category').cat.codes
    return data

def train_location_change_model(data):
    data = preprocess_data(data)
    X = data[['lat', 'lng', 'price_range', 'category', 'rating', 'reviews']]
    y = data['location_change']  # Assuming 'location_change' is a binary column indicating location change

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model Accuracy: {accuracy}")

    return model

def predict_location_change(model, data):
    data = preprocess_data(data)
    X = data[['lat', 'lng', 'price_range', 'category', 'rating', 'reviews']]
    predictions = model.predict(X)
    data['location_change_prediction'] = predictions
    return data