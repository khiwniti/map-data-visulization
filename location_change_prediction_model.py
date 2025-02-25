
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

import xgboost as xgb
from datetime import datetime
import json
import ast
import joblib
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
# Install shap if needed: pip install shap
import shap
import logging
from scipy.stats import ks_2samp
from sklearn.calibration import calibration_curve

# Configure logger
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('location_change_model')

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
        self.model_metadata = {
            'version': '1.0',
            'created_at': None,
            'features': None
        }

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
            
        # Store feature names for later use
        self.model_metadata['features'] = feature_columns

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
            logger.info(f"Training {name}...")
            
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
            
            logger.info(f"{name} ROC-AUC Score: {score:.4f}")
            logger.info(f"Classification Report:\n{classification_report(y_test, y_pred)}")
            
            if score > best_score:
                best_score = score
                best_model_name = name
                self.best_model = grid_search.best_estimator_

        logger.info(f"Best performing model: {best_model_name} (ROC-AUC: {best_score:.4f})")
        
        # Calculate feature importance
        if hasattr(self.best_model, 'feature_importances_'):
            self.feature_importance = pd.DataFrame({
                'feature': X.columns,
                'importance': self.best_model.feature_importances_
            }).sort_values('importance', ascending=False)
            logger.info(f"Feature Importance:\n{self.feature_importance}")

        self.is_trained = True
        self.model_metadata['created_at'] = datetime.now().isoformat()
        
        # Monitor initial model performance
        self.monitor_model_performance(X_test, y_test)

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

    def explain_predictions(self, X, top_n=10):
        """
        Explain model predictions using feature importance or SHAP values.
        
        Args:
            X: Feature matrix to explain predictions for
            top_n: Number of top features to include in explanation
            
        Returns:
            Dictionary with feature importance and SHAP values if available
        """
        if not self.is_trained:
            raise Exception("Model must be trained before explaining predictions.")
            
        if not hasattr(self.best_model, 'feature_importances_'):
            logger.warning("Model doesn't support feature importance explanation")
            return None
            
        # Get feature names
        feature_names = self.model_metadata['features']
        
        # Basic feature importance
        feature_importance = pd.DataFrame({
            'feature': feature_names,
            'importance': self.best_model.feature_importances_
        }).sort_values('importance', ascending=False).head(top_n)
        
        explanation = {'feature_importance': feature_importance}
        
        # Try to use SHAP for XGBoost models
        if isinstance(self.best_model, xgb.XGBClassifier):
            try:
                # Scale features if needed
                X_scaled = self.scaler.transform(X)
                
                # Create explainer
                explainer = shap.Explainer(self.best_model)
                shap_values = explainer(X_scaled)
                
                # Get mean absolute SHAP values
                mean_shap = np.abs(shap_values.values).mean(0)
                shap_importance = pd.DataFrame({
                    'feature': feature_names,
                    'shap_value': mean_shap
                }).sort_values('shap_value', ascending=False).head(top_n)
                
                explanation['shap_values'] = shap_importance
                explanation['shap_data'] = shap_values
                
                logger.info("SHAP explanation generated successfully")
            except Exception as e:
                logger.warning(f"Failed to generate SHAP explanation: {str(e)}")
        
        return explanation

    def get_prediction_insights(self, X):
        """
        Generate actionable insights based on predictions.
        
        Args:
            X: Feature matrix to make predictions on
            location_id: Optional ID to filter results for a specific location
            
        Returns:
            Dictionary with predictions and insights
        """
        if not self.is_trained:
            raise Exception("Model must be trained before generating insights.")
            
        # Make predictions
        X_scaled = self.scaler.transform(X)
        predictions = self.best_model.predict(X_scaled)
        probabilities = self.best_model.predict_proba(X_scaled)[:, 1]
        
        # Get feature importance
        explanation = self.explain_predictions(X)
        
        # Generate insights
        insights = {
            'predictions': predictions.tolist(),
            'probabilities': probabilities.tolist(),
            'top_features': explanation['feature_importance'].to_dict('records') if explanation else [],
            'recommendations': []
        }
        
        # Generate recommendations based on top features
        if explanation and 'feature_importance' in explanation:
            for _, row in explanation['feature_importance'].iterrows():
                feature = row['feature']
                importance = row['importance']
                
                if importance > 0.1:  # Only consider significant features
                    if 'rating' in feature or 'review' in feature:
                        insights['recommendations'].append(
                            "Implement a customer feedback program to improve ratings and reviews"
                        )
                    elif 'traffic' in feature or 'foot' in feature:
                        insights['recommendations'].append(
                            "Analyze foot traffic patterns and optimize opening hours"
                        )
                    elif 'competitor' in feature or 'market' in feature:
                        insights['recommendations'].append(
                            "Conduct competitive analysis and differentiate your offerings"
                        )
                    elif 'social' in feature or 'sentiment' in feature:
                        insights['recommendations'].append(
                            "Enhance social media presence and monitor online sentiment"
                        )
                    elif 'accessibility' in feature or 'transport' in feature:
                        insights['recommendations'].append(
                            "Improve location accessibility through better signage or transport options"
                        )
        
        # Remove duplicate recommendations
        insights['recommendations'] = list(set(insights['recommendations']))
        
        return insights

    def save_model(self, path):
        """
        Save the trained model to disk.
        
        Args:
            path: Path to save the model
        """
        if not self.is_trained:
            raise Exception("Model must be trained before saving.")
            
        # Create directory if it doesn't exist
        model_dir = Path(path).parent
        model_dir.mkdir(parents=True, exist_ok=True)
        
        # Prepare model data
        model_data = {
            'model': self.best_model,
            'scaler': self.scaler,
            'feature_importance': self.feature_importance,
            'metadata': {
                **self.model_metadata,
                'saved_at': datetime.now().isoformat()
            }
        }
        
        # Save model
        joblib.dump(model_data, path)
        logger.info(f"Model saved to {path}")

    def load_model(self, path):
        """
        Load a trained model from disk.
        
        Args:
            path: Path to the saved model
            
        Returns:
            self for method chaining
        """
        model_path = Path(path)
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {path}")
            
        # Load model data
        model_data = joblib.load(path)
        
        # Set model attributes
        self.best_model = model_data['model']
        self.scaler = model_data['scaler']
        self.feature_importance = model_data['feature_importance']
        self.model_metadata = model_data['metadata']
        self.is_trained = True
        
        logger.info(f"Model loaded from {path}")
        return self

    def monitor_model_performance(self, X_test, y_test, log_file=None):
        """
        Monitor model performance over time and log metrics.
        
        Args:
            X_test: Test features DataFrame
            y_test: Test target values
            log_file: Optional path to log performance metrics
            
        Returns:
            Dictionary with performance metrics
        """
        if not self.is_trained:
            logger.error("Model not trained yet")
            raise RuntimeError("Model must be trained before monitoring performance")
        
        # Scale features if scaler is available
        if self.scaler:
            X_test_scaled = self.scaler.transform(X_test)
        else:
            X_test_scaled = X_test
        
        # Make predictions
        y_pred = self.best_model.predict(X_test_scaled)
        y_pred_proba = self.best_model.predict_proba(X_test_scaled)[:, 1]
        
        # Calculate performance metrics
        metrics = {
            'timestamp': datetime.now().isoformat(),
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred, zero_division=0),
            'f1': f1_score(y_test, y_pred, zero_division=0),
            'roc_auc': roc_auc_score(y_test, y_pred_proba),
            'data_size': len(X_test)
        }
        
        # Calculate confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        metrics['confusion_matrix'] = cm.tolist()
        
        # Log metrics
        logger.info(f"Model performance: Accuracy={metrics['accuracy']:.4f}, "
                    f"Precision={metrics['precision']:.4f}, "
                    f"Recall={metrics['recall']:.4f}, "
                    f"F1={metrics['f1']:.4f}, "
                    f"ROC AUC={metrics['roc_auc']:.4f}")
        
        # Save metrics to file if specified
        if log_file:
            log_path = Path(log_file)
            
            # Create or append to log file
            if log_path.exists():
                with open(log_path, 'r') as f:
                    try:
                        log_data = json.load(f)
                    except json.JSONDecodeError:
                        log_data = {'performance_history': []}
            else:
                log_data = {'performance_history': []}
            
            # Add current metrics
            log_data['performance_history'].append(metrics)
            
            # Save updated log
            with open(log_path, 'w') as f:
                json.dump(log_data, f, indent=2)
            
            logger.info(f"Performance metrics logged to {log_file}")
        
        # Visualize confusion matrix
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['No Change', 'Location Change'],
                    yticklabels=['No Change', 'Location Change'])
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title('Confusion Matrix')
        
        # Check for performance drift
        if log_file and len(log_data['performance_history']) > 1:
            # Compare with previous performance
            prev_metrics = log_data['performance_history'][-2]
            current_metrics = metrics
            
            # Calculate drift
            drift = {
                'accuracy_drift': current_metrics['accuracy'] - prev_metrics['accuracy'],
                'precision_drift': current_metrics['precision'] - prev_metrics['precision'],
                'recall_drift': current_metrics['recall'] - prev_metrics['recall'],
                'f1_drift': current_metrics['f1'] - prev_metrics['f1'],
                'roc_auc_drift': current_metrics['roc_auc'] - prev_metrics['roc_auc']
            }
            
            # Add drift to metrics
            metrics['drift'] = drift
            
            # Log significant drift
            significant_drift = any(abs(v) > 0.05 for v in drift.values())
            if significant_drift:
                logger.warning(f"Significant performance drift detected: {drift}")
                
                # Plot performance history
                history = log_data['performance_history']
                timestamps = [h['timestamp'].split('T')[0] for h in history]
                
                plt.figure(figsize=(10, 6))
                plt.plot(timestamps, [h['accuracy'] for h in history], label='Accuracy')
                plt.plot(timestamps, [h['precision'] for h in history], label='Precision')
                plt.plot(timestamps, [h['recall'] for h in history], label='Recall')
                plt.plot(timestamps, [h['f1'] for h in history], label='F1')
                plt.plot(timestamps, [h['roc_auc'] for h in history], label='ROC AUC')
                plt.xlabel('Date')
                plt.ylabel('Score')
                plt.title('Model Performance Over Time')
                plt.legend()
                plt.xticks(rotation=45)
                plt.tight_layout()
        
        # Feature distribution monitoring
        feature_drift = self._check_feature_drift(X_test)
        if feature_drift['has_drift']:
            logger.warning(f"Feature drift detected in {len(feature_drift['drifted_features'])} features")
            metrics['feature_drift'] = feature_drift
            
            # Visualize feature drift for top drifted features
            if feature_drift['drifted_features']:
                top_drifted = feature_drift['drifted_features'][:min(3, len(feature_drift['drifted_features']))]
                plt.figure(figsize=(12, 4 * len(top_drifted)))
                
                for i, feature in enumerate(top_drifted):
                    plt.subplot(len(top_drifted), 1, i+1)
                    plt.hist(feature_drift['current_distribution'][feature], alpha=0.5, label='Current')
                    plt.hist(feature_drift['baseline_distribution'][feature], alpha=0.5, label='Baseline')
                    plt.title(f"Distribution Drift: {feature}")
                    plt.legend()
                
                plt.tight_layout()
        
        # Prediction distribution monitoring
        plt.figure(figsize=(8, 6))
        plt.hist(y_pred_proba, bins=20, alpha=0.7)
        plt.title('Prediction Probability Distribution')
        plt.xlabel('Probability of Location Change')
        plt.ylabel('Count')
        plt.axvline(x=0.5, color='red', linestyle='--', label='Decision Threshold')
        plt.legend()
        
        # Calculate calibration curve
        prob_true, prob_pred = calibration_curve(y_test, y_pred_proba, n_bins=10)
        
        plt.figure(figsize=(8, 6))
        plt.plot(prob_pred, prob_true, marker='o', label='Model')
        plt.plot([0, 1], [0, 1], linestyle='--', label='Perfectly Calibrated')
        plt.xlabel('Mean Predicted Probability')
        plt.ylabel('Fraction of Positives')
        plt.title('Calibration Curve')
        plt.legend()
        
        # Add calibration metrics
        metrics['calibration'] = {
            'expected_calibration_error': np.mean(np.abs(prob_true - prob_pred)),
            'calibration_curve': {
                'prob_true': prob_true.tolist(),
                'prob_pred': prob_pred.tolist()
            }
        }
        
        return metrics

    def _check_feature_drift(self, X_current, baseline_data=None, threshold=0.1):
        """
        Check for drift in feature distributions.
        
        Args:
            X_current: Current feature data
            baseline_data: Baseline data to compare against (uses training data if None)
            threshold: KS statistic threshold to consider as drift
        
        Returns:
            Dictionary with drift information
        """
        # If no baseline provided, use stored baseline if available
        if baseline_data is None:
            if hasattr(self, 'baseline_features'):
                baseline_data = self.baseline_features
            else:
                logger.warning("No baseline data available for drift detection")
                return {'has_drift': False, 'drifted_features': []}
        
        drift_results = {
            'has_drift': False,
            'drifted_features': [],
            'drift_scores': {},
            'current_distribution': {},
            'baseline_distribution': {}
        }
        
        # Check each feature for distribution drift using Kolmogorov-Smirnov test
        for feature in X_current.columns:
            if feature in baseline_data.columns:
                # Get current and baseline distributions
                current_values = X_current[feature].dropna().values
                baseline_values = baseline_data[feature].dropna().values
                
                # Store distributions for visualization
                drift_results['current_distribution'][feature] = current_values
                drift_results['baseline_distribution'][feature] = baseline_values
                
                # Skip if not enough data
                if len(current_values) < 10 or len(baseline_values) < 10:
                    continue
                    
                # Calculate KS statistic
                ks_stat, p_value = ks_2samp(current_values, baseline_values)
                drift_results['drift_scores'][feature] = {
                    'ks_statistic': ks_stat,
                    'p_value': p_value
                }
                
                # Check if drift detected
                if ks_stat > threshold and p_value < 0.05:
                    drift_results['drifted_features'].append(feature)
                    drift_results['has_drift'] = True
        
        # Sort drifted features by KS statistic
        if drift_results['drifted_features']:
            drift_results['drifted_features'] = sorted(
                drift_results['drifted_features'],
                key=lambda f: drift_results['drift_scores'][f]['ks_statistic'],
                reverse=True
            )
        
        return drift_results

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
        
        # Get insights
        # Add your insights code here
        
    except Exception as e:
        logger.error(f"Error occurred: {str(e)}")
