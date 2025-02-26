"""
BiteBase Restaurant Analytics - Main Application
"""

import pandas as pd
import numpy as np
from pathlib import Path
import streamlit as st
from models import CombinedLocationModel
from utils.mock_data_generator import (
    generate_mock_data,
    generate_location_data,
    generate_time_series_data,
    generate_menu_data
)

def main():
    """
    Main entry point for the BiteBase analytics system.
    Demonstrates the full pipeline from data generation to predictions.
    """
    print("Initializing BiteBase Analytics System...")
    
    # Generate sample data
    print("\nGenerating mock data...")
    location_data = generate_location_data(n_locations=100)
    menu_data = generate_menu_data(n_items=50)
    time_series = generate_time_series_data(days=365)
    features_data = generate_mock_data(n_samples=100)
    
    print(f"Generated data shapes:")
    print(f"Location data: {location_data.shape}")
    print(f"Menu data: {menu_data.shape}")
    print(f"Time series data: {time_series.shape}")
    print(f"Features data: {features_data.shape}")
    
    # Initialize the combined model
    print("\nInitializing prediction model...")
    model_path = Path(__file__).parent / "models" / "saved_model.pkl"
    model = CombinedLocationModel(str(model_path))
    
    # Make sample predictions
    print("\nMaking sample predictions...")
    sample_location = location_data.iloc[0]
    sample_features = features_data.iloc[0].to_dict()
    
    prediction = model.predict_for_location(
        lat=sample_location['lat'],
        lng=sample_location['lng'],
        current_features=sample_features
    )
    
    print("\nPrediction Results:")
    print(f"Location: ({sample_location['lat']:.4f}, {sample_location['lng']:.4f})")
    print(f"Prediction: {'Change' if prediction['prediction'] else 'Stay'}")
    print(f"Probability: {prediction['probability']:.2f}")
    
    # Show insights
    print("\nLocation Insights:")
    for rec in prediction['insights']['recommendations']:
        print(f"- {rec}")
    
    print("\nTrend Analysis:")
    for period, trend in prediction['trends'].items():
        if isinstance(trend, dict):
            print(f"{period}: {trend.get('direction', 'N/A')} "
                  f"(value: {trend.get('value', 'N/A'):.3f})")
    
    # Launch dashboard
    print("\nLaunching dashboard...")
    print("Run 'streamlit run src/restaurant_dashboard.py' to start the dashboard")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Error: {str(e)}")
        raise