import streamlit as st
from streamlit_keplergl import keplergl_static
from keplergl import KeplerGl
import pandas as pd
import numpy as np
import json
from pathlib import Path
from typing import List, Dict, Optional
from dataclasses import dataclass
import logging
from functools import lru_cache
import pygwalker as pyg
import streamlit.components.v1 as components
from datetime import datetime, timedelta

# Add project root to Python path
import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# Local imports
from src.models import CombinedLocationModel
from src.data import RAW_DATA_PATH, DYNAMIC_DATA_PATH

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
@dataclass
class MapConfig:
    """Map configuration constants"""
    DEFAULT_CENTER: List[float] = (13.7563, 100.5018)  # Bangkok
    DEFAULT_ZOOM: int = 13
    DEFAULT_RADIUS: int = 5  # km
    MIN_DISTANCE: int = 1
    MAX_DISTANCE: int = 20

@dataclass
class FileConfig:
    """File paths and related constants"""
    RESTAURANT_DATA_PATH: Path = DYNAMIC_DATA_PATH / "LMWN/restaurants.json"
    MODEL_PATH: Path = project_root / "src/models/saved_model.pkl"

class RestaurantData:
    """Handles restaurant data loading and processing"""
    
    @staticmethod
    @lru_cache(maxsize=1)
    def load_data() -> pd.DataFrame:
        """Load and process restaurant data from JSON file with caching."""
        try:
            with open(FileConfig.RESTAURANT_DATA_PATH, 'r', encoding='utf-8') as f:
                data = json.load(f)

            restaurants = []
            for restaurant in data['page']['entities']:
                try:
                    if restaurant.get('lat') and restaurant.get('lng'):
                        categories = RestaurantData._process_categories(restaurant.get('categories', []))
                        restaurants.append(RestaurantData._create_restaurant_dict(restaurant, categories))
                except Exception as e:
                    logger.error(f"Error processing restaurant: {str(e)}")
                    continue

            df = pd.DataFrame(restaurants)
            df['location'] = df.apply(lambda row: [row['lng'], row['lat']], axis=1)
            return df

        except Exception as e:
            logger.error(f"Error loading restaurant data: {str(e)}")
            raise

    @staticmethod
    def _process_categories(categories: List[Dict]) -> List[str]:
        """Process restaurant categories."""
        processed = []
        for cat in categories:
            thai_name = cat.get('name', '')
            int_name = cat.get('internationalName', '')
            processed.append(f"{thai_name} ({int_name})" if int_name else thai_name)
        return processed

    @staticmethod
    def _create_restaurant_dict(restaurant: Dict, categories: List[str]) -> Dict:
        """Create a standardized restaurant dictionary."""
        return {
            'name': restaurant.get('name', ''),
            'lat': restaurant['lat'],
            'lng': restaurant['lng'],
            'categories': categories,
            'address': restaurant.get('contact', {}).get('address', {}).get('street', ''),
            'phone': restaurant.get('contact', {}).get('phoneno', ''),
            'price_range': restaurant.get('priceRange', {}).get('name', 'N/A')
        }

class LocationAnalyzer:
    """Handles location analysis and predictions"""
    
    def __init__(self):
        self.model = CombinedLocationModel(str(FileConfig.MODEL_PATH))
        self.current_analysis = {}

    def analyze_location(self, lat: float, lng: float, context_data: Dict) -> Dict:
        """Analyze a location with real-time predictions."""
        try:
            prediction = self.model.predict_for_location(lat, lng, context_data)
            
            if prediction:
                self.current_analysis[prediction['location_id']] = prediction
                return {
                    'success': True,
                    'data': prediction
                }
            
            return {
                'success': False,
                'error': "No prediction available"
            }
            
        except Exception as e:
            logger.error(f"Error analyzing location: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }

    def get_historical_trends(self, lat: float, lng: float) -> Dict:
        """Get historical trend analysis for a location."""
        location_id = f"{lat},{lng}"
        try:
            return self.model.get_historical_data(location_id)
        except Exception as e:
            logger.error(f"Error getting historical trends: {str(e)}")
            return {}

class DashboardUI:
    """Manages the dashboard user interface"""
    
    def __init__(self):
        self.location_analyzer = LocationAnalyzer()
        self.map_instance = KeplerGl(height=600)

    def show_analysis_sidebar(self) -> Dict:
        """Show and handle sidebar controls."""
        st.sidebar.title("Analysis Controls")
        
        time_range = st.sidebar.selectbox(
            "Time Range",
            ["Last 7 days", "Last 30 days", "Last 90 days"]
        )
        
        metrics = st.sidebar.multiselect(
            "Key Metrics",
            ["Revenue", "Foot Traffic", "Competition", "Cost"],
            default=["Revenue", "Foot Traffic"]
        )
        
        threshold = st.sidebar.slider(
            "Risk Threshold",
            0.0, 1.0, 0.7,
            help="Threshold for high-risk predictions"
        )
        
        return {
            'time_range': time_range,
            'metrics': metrics,
            'threshold': threshold
        }

    def show_location_analysis(self, lat: float, lng: float, context_data: Dict):
        """Display location analysis results."""
        analysis = self.location_analyzer.analyze_location(lat, lng, context_data)
        
        if analysis['success']:
            data = analysis['data']
            
            # Display prediction summary
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric(
                    "Risk Score",
                    f"{data['probability']:.2f}",
                    delta="High Risk" if data['probability'] > 0.7 else "Low Risk"
                )
            
            with col2:
                st.metric(
                    "Trend",
                    data['trends'].get('short_term', {}).get('direction', 'stable'),
                    delta=data['trends'].get('short_term', {}).get('value', 0)
                )
            
            with col3:
                st.metric(
                    "Recommendations",
                    len(data['insights']['recommendations'])
                )
            
            # Show detailed insights
            st.subheader("Location Insights")
            tabs = st.tabs(["Trends", "Features", "Recommendations"])
            
            with tabs[0]:
                for period, trend in data['trends'].items():
                    if trend:
                        st.write(f"**{period.replace('_', ' ').title()}:**")
