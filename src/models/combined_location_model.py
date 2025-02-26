import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Optional, Any, List
import logging
from pathlib import Path
import joblib
import shap
from collections import deque
from .location_change_model import LocationChangeModel
from .realtime_location_change_model import RealtimeLocationChangeModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CombinedLocationModel:
    """
    Combines static and real-time location change prediction capabilities.
    Integrates LocationChangeModel for base predictions and 
    RealtimeLocationChangeModel for dynamic updates.
    """
    
    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize the combined model.
        
        Args:
            model_path: Optional path to load a pre-trained model
        """
        self.base_model = LocationChangeModel()
        if model_path:
            self.load_model(model_path)
        
        self.realtime_model = RealtimeLocationChangeModel(base_model=self.base_model)
        self.current_predictions = {}
        self.cached_features = {}

    def predict_for_location(self, 
                           lat: float, 
                           lng: float, 
                           current_features: Dict[str, Any]) -> Dict[str, Any]:
        """
        Make predictions for a specific location with real-time updates.
        
        Args:
            lat: Latitude of the location
            lng: Longitude of the location
            current_features: Current feature values for the location
            
        Returns:
            Dict containing predictions and insights
        """
        location_id = f"{lat},{lng}"
        
        # Update location features with coordinates
        features = self._prepare_location_features(lat, lng, current_features)
        
        # Get real-time prediction
        prediction = self.realtime_model.process_realtime_data(location_id, features)
        
        if prediction:
            self.current_predictions[location_id] = prediction
            self.cached_features[location_id] = features
            
            return {
                'location_id': location_id,
                'coordinates': {'lat': lat, 'lng': lng},
                'prediction': prediction['prediction'],
                'probability': prediction['current_probability'],
                'insights': prediction['insights'],
                'timestamp': prediction['timestamp'],
                'feature_importance': self._get_feature_importance(features),
                'trends': self._get_location_trends(location_id)
            }
        
        # If too soon for new prediction, return cached result
        return self.current_predictions.get(location_id)

    def _prepare_location_features(self, 
                                 lat: float, 
                                 lng: float, 
                                 features: Dict[str, Any]) -> Dict[str, Any]:
        """
        Prepare features for a location, including derived features.
        
        Args:
            lat: Latitude
            lng: Longitude
            features: Base features
            
        Returns:
            Dict of prepared features
        """
        prepared_features = features.copy()
        
        # Add time-based features
        current_time = datetime.now()
        prepared_features.update({
            'hour_of_day': current_time.hour,
            'day_of_week': current_time.weekday(),
            'is_weekend': 1 if current_time.weekday() >= 5 else 0,
            'lat': lat,
            'lng': lng
        })
        
        # Add time-based multipliers
        time_multiplier = self.realtime_model._get_time_multiplier(current_time.hour)
        day_multiplier = self.realtime_model._get_day_multiplier(current_time.weekday())
        
        # Adjust relevant metrics
        for metric in ['foot_traffic', 'customer_traffic']:
            if metric in prepared_features:
                prepared_features[metric] *= (time_multiplier * day_multiplier)
        
        return prepared_features

    def _get_feature_importance(self, features: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Get feature importance for current prediction."""
        try:
            df = pd.DataFrame([features])
            importance = self.base_model.get_feature_importance()
            importance['current_value'] = df[importance['feature']].iloc[0]
            return importance.to_dict('records')
        except Exception as e:
            logger.error(f"Error calculating feature importance: {str(e)}")
            return []

    def _get_location_trends(self, location_id: str) -> Dict[str, Any]:
        """Get trend analysis for a location."""
        try:
            analysis = self.realtime_model.get_historical_analysis(location_id)
            if analysis:
                return {
                    'short_term': analysis['trend_analysis']['trends'].get('short_term', {}),
                    'medium_term': analysis['trend_analysis']['trends'].get('medium_term', {}),
                    'long_term': analysis['trend_analysis']['trends'].get('long_term', {}),
                    'recommendations': analysis['trend_analysis']['recommendations']
                }
        except Exception as e:
            logger.error(f"Error getting location trends: {str(e)}")
        return {}

    def explain_prediction(self, 
                         location_id: str, 
                         features: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Generate detailed explanation for a prediction."""
        try:
            features_to_explain = features or self.cached_features.get(location_id)
            if not features_to_explain:
                return {}
            
            df = pd.DataFrame([features_to_explain])
            shap_explainer = shap.TreeExplainer(self.base_model.model)
            shap_values = shap_explainer.shap_values(df)
            
            return {
                'shap_values': shap_values,
                'feature_names': list(df.columns),
                'base_value': float(shap_values.mean()),
                'current_prediction': self.current_predictions.get(location_id, {})
            }
            
        except Exception as e:
            logger.error(f"Error generating prediction explanation: {str(e)}")
            return {}

    def get_historical_data(self, location_id: str) -> Dict[str, Any]:
        """Get historical data for a location."""
        return self.realtime_model.get_historical_analysis(location_id)

    def load_model(self, filepath: str) -> None:
        """Load a saved model."""
        try:
            model_data = joblib.load(filepath)
            self.base_model = model_data['model']
            logger.info(f"Model loaded successfully from {filepath}")
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise

    def save_model(self, filepath: str) -> None:
        """Save the current model."""
        try:
            model_data = {
                'model': self.base_model,
                'timestamp': datetime.now().isoformat()
            }
            joblib.dump(model_data, filepath)
            logger.info(f"Model saved successfully to {filepath}")
        except Exception as e:
            logger.error(f"Error saving model: {str(e)}")
            raise

    def reset_location(self, location_id: str) -> None:
        """Reset tracking for a location."""
        if location_id in self.current_predictions:
            del self.current_predictions[location_id]
        if location_id in self.cached_features:
            del self.cached_features[location_id]
        self.realtime_model.initialize_location_tracking(location_id)