"""
Generate mock data for restaurant analytics
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional

def generate_mock_data(n_samples: int = 1000) -> pd.DataFrame:
    """
    Generate synthetic data for testing and development.
    
    Args:
        n_samples: Number of samples to generate
        
    Returns:
        DataFrame with mock restaurant data
    """
    np.random.seed(42)
    
    # Generate base features
    data = pd.DataFrame({
        'current_revenue': np.random.normal(100000, 25000, n_samples),
        'revenue_trend': np.random.uniform(-0.2, 0.3, n_samples),
        'customer_traffic': np.random.normal(1000, 200, n_samples),
        'competitor_count': np.random.poisson(5, n_samples),
        'rent_cost': np.random.normal(5000, 1000, n_samples),
        'local_population': np.random.normal(50000, 10000, n_samples),
        'avg_income': np.random.normal(60000, 15000, n_samples),
        'parking_score': np.random.uniform(0, 10, n_samples),
        'accessibility_score': np.random.uniform(0, 10, n_samples)
    })
    
    # Add derived features
    data['rent_to_revenue_ratio'] = data['rent_cost'] / data['current_revenue']
    data['revenue_per_customer'] = data['current_revenue'] / data['customer_traffic']
    
    return data

def generate_time_series_data(days: int = 365) -> pd.DataFrame:
    """
    Generate time series data for historical analysis.
    
    Args:
        days: Number of days of historical data
        
    Returns:
        DataFrame with daily metrics
    """
    dates = pd.date_range(end=datetime.now(), periods=days)
    
    data = pd.DataFrame({
        'date': dates,
        'daily_sales': np.random.normal(1000, 200, days),
        'customer_count': np.random.normal(100, 20, days),
        'avg_ticket': np.random.normal(50, 10, days),
        'weather': np.random.choice(['Sunny', 'Rainy', 'Cloudy'], days),
        'is_weekend': dates.weekday >= 5,
        'is_holiday': np.random.choice([True, False], days, p=[0.1, 0.9])
    })
    
    # Add seasonality
    data['daily_sales'] += np.sin(np.pi * data.index / 180) * 300  # Seasonal pattern
    data['customer_count'] += np.sin(np.pi * data.index / 180) * 30
    
    return data

def generate_location_data(n_locations: int = 100) -> pd.DataFrame:
    """
    Generate location-based data for spatial analysis.
    
    Args:
        n_locations: Number of locations to generate
        
    Returns:
        DataFrame with location-specific data
    """
    # Bangkok area boundaries
    lat_range = (13.6, 13.9)
    lng_range = (100.4, 100.7)
    
    data = pd.DataFrame({
        'lat': np.random.uniform(lat_range[0], lat_range[1], n_locations),
        'lng': np.random.uniform(lng_range[0], lng_range[1], n_locations),
        'location_score': np.random.uniform(0, 10, n_locations),
        'population_density': np.random.normal(5000, 1000, n_locations),
        'competitor_density': np.random.uniform(0.1, 0.9, n_locations),
        'rent_per_sqm': np.random.normal(800, 200, n_locations),
        'foot_traffic': np.random.normal(500, 100, n_locations)
    })
    
    return data

def generate_menu_data(n_items: int = 50) -> pd.DataFrame:
    """
    Generate menu item performance data.
    
    Args:
        n_items: Number of menu items
        
    Returns:
        DataFrame with menu item metrics
    """
    categories = ['Main Course', 'Appetizer', 'Dessert', 'Beverage', 'Special']
    
    data = pd.DataFrame({
        'item_name': [f'Item_{i}' for i in range(n_items)],
        'category': np.random.choice(categories, n_items),
        'price': np.random.uniform(50, 500, n_items),
        'cost': np.random.uniform(20, 200, n_items),
        'sales_volume': np.random.randint(100, 1000, n_items),
        'rating': np.random.uniform(3.5, 5.0, n_items),
        'preparation_time': np.random.uniform(5, 45, n_items)
    })
    
    # Calculate derived metrics
    data['profit_margin'] = (data['price'] - data['cost']) / data['price']
    data['daily_profit'] = (data['price'] - data['cost']) * data['sales_volume'] / 30
    
    return data

if __name__ == "__main__":
    # Example usage
    location_data = generate_location_data()
    menu_data = generate_menu_data()
    time_series = generate_time_series_data()
    
    print("Generated mock data shapes:")
    print(f"Location data: {location_data.shape}")
    print(f"Menu data: {menu_data.shape}")
    print(f"Time series data: {time_series.shape}")
