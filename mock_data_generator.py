import pandas as pd
import numpy as np
import geopandas as gpd
from shapely.geometry import Polygon, Point
from datetime import datetime, timedelta

def generate_strategic_points(num_points, bounds):
    """Generate strategic points like malls, schools, offices"""
    points = []
    types = ['mall', 'school', 'office', 'transport_hub', 'tourist_spot']
    for _ in range(num_points):
        lat = np.random.uniform(bounds[0], bounds[1])
        lon = np.random.uniform(bounds[2], bounds[3])
        point_type = np.random.choice(types)
        points.append({
            'type': point_type,
            'location': Point(lon, lat),
            'importance_score': np.random.uniform(0.5, 1.0)
        })
    return points

def generate_social_data(num_samples):
    """Generate mock social media and review data"""
    return pd.DataFrame({
        'avg_rating': np.random.uniform(3.0, 5.0, num_samples),
        'review_count': np.random.randint(10, 1000, num_samples),
        'sentiment_score': np.random.uniform(0.3, 0.9, num_samples),
        'social_media_mentions': np.random.randint(0, 500, num_samples),
        'peak_hours_traffic': np.random.randint(50, 200, num_samples),
    })

def generate_historical_data(num_samples):
    """Generate historical business performance data"""
    start_date = datetime.now() - timedelta(days=365)
    dates = [start_date + timedelta(days=x) for x in range(12)]
    
    historical_data = []
    for _ in range(num_samples):
        monthly_data = {
            'monthly_revenue': [np.random.randint(50000, 500000) for _ in range(12)],
            'customer_count': [np.random.randint(1000, 5000) for _ in range(12)],
            'competitor_count': [np.random.randint(1, 10) for _ in range(12)]
        }
        historical_data.append(monthly_data)
    return historical_data

def generate_mock_data(num_zones=10):
    np.random.seed(42)
    
    # Define bounds for Bangkok area
    lat_bounds = (13.5, 13.9)  # Bangkok latitude range
    lon_bounds = (100.4, 100.9)  # Bangkok longitude range
    
    latitudes = np.random.uniform(lat_bounds[0], lat_bounds[1], num_zones)
    longitudes = np.random.uniform(lon_bounds[0], lon_bounds[1], num_zones)
    zones = []

    # Generate zones
    for i in range(num_zones):
        lat, lon = latitudes[i], longitudes[i]
        zone = Polygon([
            (lon, lat),
            (lon + 0.02, lat),
            (lon + 0.02, lat + 0.02),
            (lon, lat + 0.02)
        ])
        zones.append(zone)

    # Generate strategic points
    strategic_points = generate_strategic_points(20, [*lat_bounds, *lon_bounds])
    
    # Generate social and review data
    social_data = generate_social_data(num_zones)
    
    # Generate historical data
    historical_data = generate_historical_data(num_zones)
    
    # Create base dataframe
    data = gpd.GeoDataFrame({
        "zone_id": range(num_zones),
        "population_density": np.random.randint(5000, 50000, num_zones),
        "real_estate_value": np.random.randint(50000, 500000, num_zones),
        "foot_traffic": np.random.randint(1000, 10000, num_zones),
        "nearby_competitors": np.random.randint(1, 20, num_zones),
        "avg_income_level": np.random.uniform(30000, 100000, num_zones),
        "parking_availability": np.random.randint(10, 100, num_zones),
        "public_transport_score": np.random.uniform(0.1, 1.0, num_zones),
        "geometry": zones
    })
    
    # Add social data
    for col in social_data.columns:
        data[col] = social_data[col]
    
    # Add strategic points influence
    data['strategic_points_score'] = 0
    for point in strategic_points:
        distances = data.geometry.apply(lambda x: x.centroid.distance(point['location']))
        influence = point['importance_score'] / (1 + distances)
        data['strategic_points_score'] += influence
    
    # Add historical data
    data['historical_data'] = historical_data
    
    # Generate target variable (location_change probability)
    data['location_change_probability'] = calculate_location_change_probability(data)
    
    return data

def calculate_location_change_probability(data):
    """Calculate probability of location change based on features"""
    probabilities = np.zeros(len(data))
    
    # Normalize and weigh different factors
    normalized_features = {
        'population_density': (data['population_density'] - data['population_density'].min()) / 
                            (data['population_density'].max() - data['population_density'].min()),
        'real_estate_value': (data['real_estate_value'] - data['real_estate_value'].min()) /
                            (data['real_estate_value'].max() - data['real_estate_value'].min()),
        'strategic_score': (data['strategic_points_score'] - data['strategic_points_score'].min()) /
                          (data['strategic_points_score'].max() - data['strategic_points_score'].min()),
        'sentiment': data['sentiment_score'],
        'traffic': (data['foot_traffic'] - data['foot_traffic'].min()) /
                  (data['foot_traffic'].max() - data['foot_traffic'].min())
    }
    
    # Assign weights to different factors
    weights = {
        'population_density': 0.2,
        'real_estate_value': 0.3,
        'strategic_score': 0.2,
        'sentiment': 0.15,
        'traffic': 0.15
    }
    
    # Calculate weighted sum
    for feature, weight in weights.items():
        probabilities += normalized_features[feature] * weight
    
    # Add some random noise
    probabilities += np.random.normal(0, 0.1, len(probabilities))
    
    # Normalize to 0-1 range
    probabilities = (probabilities - probabilities.min()) / (probabilities.max() - probabilities.min())
    
    return probabilities

if __name__ == "__main__":
    data = generate_mock_data(20)  # Generate 20 zones
    data.to_file("mock_data.geojson", driver="GeoJSON")
    
    # Also save a CSV version for easier model training
    data_csv = data.copy()
    data_csv['geometry'] = data_csv['geometry'].apply(lambda x: x.wkt)
    data_csv['historical_data'] = data_csv['historical_data'].apply(str)
    data_csv.to_csv("mock_data.csv", index=False)
    
    print("Mock data generated and saved to mock_data.geojson and mock_data.csv")
    print(f"Generated {len(data)} zones with the following features:")
    for col in data.columns:
        print(f"- {col}")
