import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import time
from collections import deque
import sys
from pathlib import Path

# Add parent directory to Python path when running as main script
if __name__ == "__main__":
    sys.path.append(str(Path(__file__).parent.parent.parent))
    from src.models.location_change_model import LocationChangeModel
else:
    from .location_change_model import LocationChangeModel

class RealtimeLocationChangeModel:
    def __init__(self, base_model=None, window_size=30):
        """
        Initialize the realtime model
        base_model: Pre-trained LocationChangeModel
        window_size: Number of days to keep in memory for trend analysis
        """
        self.base_model = base_model if base_model else LocationChangeModel()
        self.window_size = window_size
        self.historical_data = {}
        self.trend_windows = {}
        self.last_update = {}
        self.alert_thresholds = {
            'probability_change': 0.2,  # Significant change in probability
            'trend_threshold': 0.1,     # Significant trend change
            'update_frequency': 3600    # Minimum seconds between updates
        }

    def initialize_location_tracking(self, location_id):
        """Initialize tracking for a new location"""
        self.historical_data[location_id] = {
            'predictions': deque(maxlen=self.window_size),
            'features': deque(maxlen=self.window_size),
            'timestamps': deque(maxlen=self.window_size)
        }
        self.trend_windows[location_id] = {
            'short_term': deque(maxlen=7),    # 7-day trend
            'medium_term': deque(maxlen=14),   # 14-day trend
            'long_term': deque(maxlen=30)      # 30-day trend
        }
        self.last_update[location_id] = datetime.now()

    def update_realtime_features(self, location_data):
        """Update feature values based on real-time data"""
        updated_data = location_data.copy()
        
        # Update time-based features
        current_hour = datetime.now().hour
        current_day = datetime.now().weekday()
        
        # Adjust foot traffic based on time of day
        time_multiplier = self._get_time_multiplier(current_hour)
        updated_data['foot_traffic'] = updated_data.get('foot_traffic', 0) * time_multiplier
        
        # Adjust based on day of week
        day_multiplier = self._get_day_multiplier(current_day)
        updated_data['foot_traffic'] = updated_data['foot_traffic'] * day_multiplier
        
        # Update social impact based on time
        if 'social_media_mentions' in updated_data:
            updated_data['social_media_mentions'] = self._adjust_social_metrics(
                updated_data['social_media_mentions'],
                current_hour
            )
        
        return updated_data

    def _get_time_multiplier(self, hour):
        """Get multiplier for different times of day"""
        if 11 <= hour <= 14:  # Lunch rush
            return 1.5
        elif 17 <= hour <= 21:  # Dinner rush
            return 1.8
        elif 22 <= hour <= 5:  # Late night
            return 0.3
        else:  # Normal hours
            return 1.0

    def _get_day_multiplier(self, day):
        """Get multiplier for different days of week"""
        if day >= 5:  # Weekend
            return 1.4
        return 1.0

    def _adjust_social_metrics(self, value, hour):
        """Adjust social metrics based on time of day"""
        if 9 <= hour <= 22:  # Active hours
            return value * 1.2
        return value * 0.8

    def process_realtime_data(self, location_id, current_data):
        """Process incoming real-time data for a location"""
        if location_id not in self.historical_data:
            self.initialize_location_tracking(location_id)
        
        # Check if enough time has passed since last update
        current_time = datetime.now()
        if (current_time - self.last_update[location_id]).total_seconds() < self.alert_thresholds['update_frequency']:
            return None
        
        # Update features with real-time adjustments
        updated_data = self.update_realtime_features(current_data)
        
        # Make prediction
        prediction_result = self.base_model.predict(pd.DataFrame([updated_data]))
        current_probability = prediction_result['change_probability'].iloc[0]
        
        # Store results
        self.historical_data[location_id]['predictions'].append(current_probability)
        self.historical_data[location_id]['features'].append(updated_data)
        self.historical_data[location_id]['timestamps'].append(current_time)
        
        # Update trend windows
        self._update_trends(location_id, current_probability)
        
        # Generate insights
        insights = self._analyze_trends(location_id)
        
        self.last_update[location_id] = current_time
        
        return {
            'location_id': location_id,
            'current_probability': current_probability,
            'prediction': bool(prediction_result['location_change_prediction'].iloc[0]),
            'insights': insights,
            'timestamp': current_time.isoformat()
        }

    def _update_trends(self, location_id, probability):
        """Update trend windows with new probability"""
        for window in self.trend_windows[location_id].values():
            window.append(probability)

    def _analyze_trends(self, location_id):
        """Analyze trends and generate insights"""
        insights = {
            'trends': {},
            'alerts': [],
            'recommendations': []
        }
        
        # Analyze different time windows
        for period, window in self.trend_windows[location_id].items():
            if len(window) >= 2:
                trend = (window[-1] - window[0]) / len(window)
                insights['trends'][period] = {
                    'value': trend,
                    'direction': 'increasing' if trend > 0 else 'decreasing',
                    'significant': abs(trend) > self.alert_thresholds['trend_threshold']
                }
        
        # Generate alerts
        if insights['trends'].get('short_term', {}).get('significant'):
            insights['alerts'].append({
                'level': 'high',
                'message': f"Significant {insights['trends']['short_term']['direction']} trend detected in last 7 days"
            })
        
        # Add recommendations based on trends
        self._generate_recommendations(insights)
        
        return insights

    def _generate_recommendations(self, insights):
        """Generate recommendations based on insights"""
        for trend_info in insights['trends'].values():
            if trend_info.get('significant'):
                if trend_info['direction'] == 'increasing':
                    insights['recommendations'].append(
                        "Consider proactive measures to maintain positive momentum"
                    )
                else:
                    insights['recommendations'].append(
                        "Investigate potential causes of declining predictions"
                    )

    def get_historical_analysis(self, location_id):
        """Get historical analysis for a location"""
        if location_id not in self.historical_data:
            return None
        
        data = self.historical_data[location_id]
        return {
            'predictions': list(data['predictions']),
            'timestamps': [t.isoformat() for t in data['timestamps']],
            'trend_analysis': self._analyze_trends(location_id),
            'feature_evolution': self._analyze_feature_evolution(location_id)
        }

    def _analyze_feature_evolution(self, location_id):
        """Analyze how features have evolved over time"""
        features_data = self.historical_data[location_id]['features']
        if not features_data:
            return {}
        
        evolution = {}
        feature_names = features_data[0].keys()
        
        for feature in feature_names:
            values = [data[feature] for data in features_data]
            evolution[feature] = {
                'start': values[0],
                'end': values[-1],
                'change': ((values[-1] - values[0]) / values[0]) if values[0] != 0 else 0,
                'trend': 'increasing' if values[-1] > values[0] else 'decreasing'
            }
        
        return evolution

if __name__ == "__main__":
    # Example usage
    try:
        # Create a simple location model
        base_model = LocationChangeModel()
        
        # Initialize realtime model
        realtime_model = RealtimeLocationChangeModel(base_model)
        
        # Generate sample data
        sample_data = {
            'current_revenue': 100000,
            'customer_traffic': 1000,
            'competitor_count': 5,
            'rent_cost': 5000,
            'local_population': 50000,
            'avg_income': 60000,
            'parking_score': 7,
            'accessibility_score': 8,
            'foot_traffic': 500
        }
        
        # Process real-time data
        location_id = "test_location"
        result = realtime_model.process_realtime_data(location_id, sample_data)
        
        print("\nReal-time Prediction Result:")
        print(json.dumps(result, indent=2))
        
        # Wait and process another update
        time.sleep(2)
        result = realtime_model.process_realtime_data(location_id, sample_data)
        
        print("\nHistorical Analysis:")
        analysis = realtime_model.get_historical_analysis(location_id)
        print(json.dumps(analysis, indent=2))
        
    except Exception as e:
        print(f"Error in example: {str(e)}")
