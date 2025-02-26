import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from typing import Dict, Any, List
import logging

logger = logging.getLogger(__name__)

class LocationChangeModel:
    """Base model for location change predictions"""
    
    def __init__(self):
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42
        )
        self.scaler = StandardScaler()
        self.feature_names = []
        
    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare features for prediction"""
        # List of expected features
        required_features = [
            'current_revenue', 'revenue_trend', 'customer_traffic',
            'competitor_count', 'rent_cost', 'local_population',
            'avg_income', 'parking_score', 'accessibility_score'
        ]
        
        # Ensure all required features are present
        for feature in required_features:
            if feature not in df.columns:
                df[feature] = np.nan
        
        return df[required_features]
        
    def predict(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Make predictions for given data"""
        try:
            # Prepare features
            features = self.prepare_features(df)
            
            # Scale features
            features_scaled = self.scaler.transform(features)
            
            # Make prediction
            predictions = self.model.predict(features_scaled)
            probabilities = self.model.predict_proba(features_scaled)
            
            # Prepare results
            results = pd.DataFrame({
                'location_change_prediction': predictions,
                'change_probability': probabilities[:, 1]
            })
            
            if 'name' in df.columns:
                results['name'] = df['name']
            
            return results
            
        except Exception as e:
            logger.error(f"Error making prediction: {str(e)}")
            raise
            
    def get_feature_importance(self) -> pd.DataFrame:
        """Get feature importance scores"""
        try:
            importance = pd.DataFrame({
                'feature': self.feature_names,
                'importance': self.model.feature_importances_
            })
            return importance.sort_values('importance', ascending=False)
        except Exception as e:
            logger.error(f"Error getting feature importance: {str(e)}")
            raise