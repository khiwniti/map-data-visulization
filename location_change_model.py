import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.pipeline import Pipeline
import xgboost as xgb
from datetime import datetime
import json
import ast

class LocationChangeModel:
    def __init__(self):
        self.models = {
            'rf': RandomForestClassifier(random_state=42),
            'gb': GradientBoostingClassifier(random_state=42),
            'xgb': xgb.XGBClassifier(random_state=42)
        }
        self.best_model = None
        self.feature_importance = None
        self.scaler = StandardScaler()
        self.is_trained = False

    def preprocess_data(self, data):
        """Preprocess and engineer features from raw data"""
        processed_data = data.copy()

        # Handle historical data
        if 'historical_data' in processed_data.columns:
            processed_data['historical_data'] = processed_data['historical_data'].apply(
                lambda x: ast.literal_eval(x) if isinstance(x, str) else x
            )
            # Extract historical metrics
            processed_data['avg_monthly_revenue'] = processed_data['historical_data'].apply(
                lambda x: np.mean(x['monthly_revenue'])
            )
            processed_data['revenue_trend'] = processed_data['historical_data'].apply(
                lambda x: (x['monthly_revenue'][-1] - x['monthly_revenue'][0]) / x['monthly_revenue'][0]
            )
            processed_data['avg_customer_count'] = processed_data['historical_data'].apply(
                lambda x: np.mean(x['customer_count'])
            )
            processed_data['competitor_trend'] = processed_data['historical_data'].apply(
                lambda x: (x['competitor_count'][-1] - x['competitor_count'][0]) / max(x['competitor_count'][0], 1)
            )

        # Strategic location score
        if 'strategic_points_score' in processed_data.columns:
            processed_data['strategic_importance'] = processed_data['strategic_points_score'] * \
                                                   processed_data['public_transport_score']

        # Competitive pressure
        if 'nearby_competitors' in processed_data.columns:
            processed_data['market_saturation'] = processed_data['nearby_competitors'] / \
                                                processed_data['population_density']

        # Economic indicators
        if 'real_estate_value' in processed_data.columns and 'avg_income_level' in processed_data.columns:
            processed_data['affordability_ratio'] = processed_data['avg_income_level'] / \
                                                  processed_data['real_estate_value']

        # Social sentiment impact
        if 'sentiment_score' in processed_data.columns and 'social_media_mentions' in processed_data.columns:
            processed_data['social_impact'] = processed_data['sentiment_score'] * \
                                            np.log1p(processed_data['social_media_mentions'])

        # Location accessibility
        if 'parking_availability' in processed_data.columns and 'public_transport_score' in processed_data.columns:
            processed_data['accessibility_score'] = (processed_data['parking_availability'] / 100 + 
                                                   processed_data['public_transport_score']) / 2

        # Drop non-numeric and unnecessary columns
        columns_to_drop = ['historical_data', 'geometry']
        processed_data = processed_data.drop(columns=[col for col in columns_to_drop if col in processed_data.columns])

        return processed_data

    def prepare_features(self, data):
        """Prepare feature matrix X and target variable y"""
        processed_data = self.preprocess_data(data)
        
        # Select features for training
        feature_columns = [
            'population_density', 'real_estate_value', 'foot_traffic',
            'nearby_competitors', 'avg_income_level', 'parking_availability',
            'public_transport_score', 'avg_rating', 'review_count',
            'sentiment_score', 'social_media_mentions', 'peak_hours_traffic',
            'strategic_points_score', 'strategic_importance', 'market_saturation',
            'affordability_ratio', 'social_impact', 'accessibility_score'
        ]

        # Only use available columns
        feature_columns = [col for col in feature_columns if col in processed_data.columns]
        
        X = processed_data[feature_columns]
        y = (processed_data['location_change_probability'] > 0.5).astype(int) \
            if 'location_change_probability' in processed_data.columns \
            else processed_data['location_change']

        return X, y

    def train_model(self, data):
        """Train multiple models and select the best performing one"""
        X, y = self.prepare_features(data)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        best_score = 0
        best_model_name = None

        # Train and evaluate each model
        for name, model in self.models.items():
            print(f"\nTraining {name}...")
            
            # Define parameter grid for GridSearchCV
            param_grid = {
                'rf': {
                    'n_estimators': [100, 200],
                    'max_depth': [10, 20, None],
                    'min_samples_split': [2, 5]
                },
                'gb': {
                    'n_estimators': [100, 200],
                    'learning_rate': [0.01, 0.1],
                    'max_depth': [3, 5]
                },
                'xgb': {
                    'n_estimators': [100, 200],
                    'max_depth': [3, 5],
                    'learning_rate': [0.01, 0.1]
                }
            }

            # Perform GridSearchCV
            grid_search = GridSearchCV(model, param_grid[name], cv=5, scoring='roc_auc')
            grid_search.fit(X_train_scaled, y_train)
            
            # Make predictions
            y_pred = grid_search.predict(X_test_scaled)
            score = roc_auc_score(y_test, y_pred)
            
            print(f"{name} ROC-AUC Score: {score:.4f}")
            print("Classification Report:")
            print(classification_report(y_test, y_pred))
            
            if score > best_score:
                best_score = score
                best_model_name = name
                self.best_model = grid_search.best_estimator_

        print(f"\nBest performing model: {best_model_name} (ROC-AUC: {best_score:.4f})")
        
        # Calculate feature importance
        if hasattr(self.best_model, 'feature_importances_'):
            self.feature_importance = pd.DataFrame({
                'feature': X.columns,
                'importance': self.best_model.feature_importances_
            }).sort_values('importance', ascending=False)
            print("\nFeature Importance:")
            print(self.feature_importance)

        self.is_trained = True

    def predict(self, data):
        """Make predictions using the best trained model"""
        if not self.is_trained:
            raise Exception("Model is not trained yet. Please train the model before making predictions.")
        
        X, _ = self.prepare_features(data)
        X_scaled = self.scaler.transform(X)
        predictions = self.best_model.predict(X_scaled)
        probabilities = self.best_model.predict_proba(X_scaled)[:, 1]
        
        result = data.copy()
        result['location_change_prediction'] = predictions
        result['change_probability'] = probabilities
        return result

    def get_location_insights(self, data, location_id):
        """Get detailed insights for a specific location"""
        if not self.is_trained or self.feature_importance is None:
            raise Exception("Model must be trained before getting insights.")
        
        location_data = data[data['zone_id'] == location_id].iloc[0]
        processed_data = self.preprocess_data(pd.DataFrame([location_data]))
        
        insights = {
            'top_factors': [],
            'recommendations': []
        }
        
        # Analyze top influencing factors
        for _, row in self.feature_importance.head().iterrows():
            feature = row['feature']
            if feature in processed_data.columns:
                value = processed_data[feature].iloc[0]
                insights['top_factors'].append({
                    'factor': feature,
                    'importance': row['importance'],
                    'value': value
                })
        
        # Generate recommendations based on the most important features
        for factor in insights['top_factors']:
            if factor['importance'] > 0.1:  # Only consider significant factors
                recommendation = self.generate_recommendation(factor)
                if recommendation:
                    insights['recommendations'].append(recommendation)
        
        return insights

    def generate_recommendation(self, factor):
        """Generate specific recommendations based on factor analysis"""
        recommendations = {
            'population_density': "Consider demographic trends and local community needs",
            'real_estate_value': "Analyze property value trends and rental costs",
            'foot_traffic': "Evaluate peak hours and customer flow patterns",
            'social_impact': "Focus on improving online presence and customer engagement",
            'accessibility_score': "Consider improving parking or public transport accessibility",
            'market_saturation': "Analyze competitor landscape and market positioning"
        }
        return recommendations.get(factor['factor'])

if __name__ == "__main__":
    # Load mock data
    try:
        data = pd.read_csv("mock_data.csv")
        
        # Initialize and train the model
        model = LocationChangeModel()
        model.train_model(data)
        
        # Make predictions
        predictions = model.predict(data)
        print("\nSample Predictions:")
        print(predictions[['zone_id', 'location_change_prediction', 'change_probability']].head())
        
        # Get insights for a specific location
        location_id = 0
        insights = model.get_location_insights(data, location_id)
        print(f"\nInsights for location {location_id}:")
        print(json.dumps(insights, indent=2))
        
    except Exception as e:
        print(f"Error: {str(e)}")
