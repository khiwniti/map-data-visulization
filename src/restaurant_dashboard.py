import streamlit as st
import pandas as pd
# import numpy as np
# import json
from pathlib import Path
from typing import List, Dict
from dataclasses import dataclass, field
import logging
from functools import lru_cache
# import pygwalker as pyg
# import streamlit.components.v1 as components
# import radar  # Import the radar package
import folium  # We'll use folium as an alternative map visualization
from streamlit_folium import folium_static  # For displaying folium maps in Streamlit

# Add project root to Python path
import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# Configure logging with more details
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Print debug information
logger.debug(f"Project root: {project_root}")

# Local imports
try:
    logger.debug("Attempting to import models...")
    from src.models import CombinedLocationModel
    logger.debug("Successfully imported CombinedLocationModel")
    
    logger.debug("Attempting to import data paths...")
    from src.data import RAW_DATA_PATH, DYNAMIC_DATA_PATH
    logger.debug(f"Data paths: RAW={RAW_DATA_PATH}, DYNAMIC={DYNAMIC_DATA_PATH}")
except Exception as e:
    logger.error(f"Import error: {str(e)}")
    st.error(f"Failed to import required modules: {str(e)}")

# Constants
@dataclass
class MapConfig:
    """Map configuration constants"""
    DEFAULT_CENTER: List[float] = field(default_factory=lambda: [13.7563, 100.5018])  # Bangkok
    DEFAULT_ZOOM: int = 13
    DEFAULT_RADIUS: int = 5  # km
    MIN_DISTANCE: int = 1
    MAX_DISTANCE: int = 20

    @staticmethod
    def get_default_center() -> List[float]:
        """Get the default center coordinates"""
        return [13.7563, 100.5018]  # Return the values directly instead of accessing class attribute
 
@dataclass
class FileConfig:
    """File paths and related constants"""
    RESTAURANT_DATA_PATH: Path = Path(__file__).parent.parent.parent / "mock_restaurant_data.csv"
    MODEL_PATH: Path = project_root / "src/models/saved_model.pkl"

class RestaurantData:
    """Handles restaurant data loading and processing"""
    
    @staticmethod
    @lru_cache(maxsize=1)
    def load_data() -> pd.DataFrame:
        """Load restaurant data from a CSV file."""
        try:
            logger.debug(f"Attempting to load CSV from: {FileConfig.RESTAURANT_DATA_PATH}")
            if not FileConfig.RESTAURANT_DATA_PATH.exists():
                raise FileNotFoundError(f"CSV file not found at {FileConfig.RESTAURANT_DATA_PATH}")
            
            df = pd.read_csv(FileConfig.RESTAURANT_DATA_PATH)
            logger.debug(f"Successfully loaded CSV with {len(df)} rows")
            
            # Basic data cleaning and transformation
            df = df.dropna(subset=['GPS_Data'])
            df[['lat', 'lng']] = df['GPS_Data'].str.strip('[]').str.split(',', expand=True).astype(float)
            
            # Use actual columns from CSV
            if 'Categories' in df.columns:
                df['categories'] = df['Categories']
            else:
                df['categories'] = 'Various'
                
            if 'Price_Range' in df.columns:
                df['price_range'] = df['Price_Range']
            else:
                df['price_range'] = 'N/A'
                
            if 'Name' in df.columns:
                df['name'] = df['Name']
            else:
                df['name'] = 'Restaurant'
                
            return df

        except FileNotFoundError as fnf:
            logger.error(f"CSV file not found: {str(fnf)}")
            st.error(f"CSV file not found. Please check if the file exists at: {FileConfig.RESTAURANT_DATA_PATH}")
            return pd.DataFrame()
        except Exception as e:
            logger.error(f"Error loading restaurant data: {str(e)}")
            st.error(f"An unexpected error occurred: {str(e)}")
            return pd.DataFrame()

class LocationAnalyzer:
    """Handles location analysis and predictions"""
    
    def __init__(self):
        self.model = CombinedLocationModel()
        self.current_analysis = {}

    def analyze_location(self, lat: float, lng: float, context_data: Dict) -> Dict:
        """Analyze a location with real-time predictions."""
        try:
            prediction = self.model.predict_for_location(lat, lng, current_features=context_data)
            
            if prediction:
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
            return self.model.get_historical_data(location_id) or {}
        except Exception as e:
            logger.error(f"Error getting historical trends: {str(e)}")
            return self.model.get_historical_data(location_id)

class RadarMapUI:
    """Manages the Radar map user interface"""
    
    def __init__(self):
        try:
            self.api_key = st.secrets["radar_api_key"]
        except (KeyError, FileNotFoundError):
            logger.warning("Radar API key not found in secrets. Using default configuration.")
            self.api_key = "prj_test_sk_4ad0c4b687568f5cca83d3acc2c808fefe9"  # Default test key
            st.warning("Using default map configuration. Some features may be limited.")
    
    def create_map(self, center_lat, center_lng, zoom=13):
        """Create a Folium map with Radar integration"""
        # Create a folium map
        m = folium.Map(location=[center_lat, center_lng], zoom_start=zoom, 
                      tiles="CartoDB positron")
        
        # Add a marker for the center location
        folium.Marker(
            [center_lat, center_lng],
            popup="Selected Location",
            icon=folium.Icon(color="red", icon="info-sign"),
        ).add_to(m)
        
        return m
    
    def add_restaurants_to_map(self, m, restaurants_df):
        """Add restaurant markers to the map"""
        # Create a feature group for restaurants
        restaurant_group = folium.FeatureGroup(name="Restaurants")
        
        for idx, row in restaurants_df.iterrows():
            folium.Marker(
                [row['lat'], row['lng']],
                popup=f"<b>{row['name']}</b><br>Categories: {row['categories']}<br>Price: {row['price_range']}",
                icon=folium.Icon(color="blue", icon="cutlery"),
            ).add_to(restaurant_group)
        
        # Add the restaurant group to the map
        restaurant_group.add_to(m)
        
        return m
    
    def add_heatmap(self, m, restaurants_df):
        """Add a heatmap layer to the map"""
        # Extract coordinates for heatmap
        heat_data = [[row['lat'], row['lng']] for idx, row in restaurants_df.iterrows()]
        
        # Add heatmap layer
        from folium.plugins import HeatMap
        HeatMap(heat_data).add_to(m)
        
        return m

class DashboardUI:
    """Manages the dashboard user interface"""
    
    def __init__(self):
        self.location_analyzer = LocationAnalyzer()
        self.radar_map = RadarMapUI()

    def show_analysis_sidebar(self) -> Dict:
        """Show and handle sidebar controls."""
        st.sidebar.title("Analysis Controls")
        
        time_range = st.sidebar.selectbox(
            "Time Range",
            ["Last 7 days", "Last 30 days", "Last 90 days"]
        )
        
        metrics = st.sidebar.multiselect(
            "Category Metrics",
            ["Revenue", "Foot Traffic", "Competition", "Cost"],
            default=["Revenue", "Foot Traffic"]
        )
        
        threshold = st.sidebar.slider(
            "Radius Threshold (km)",
            0.0, 0.1, 10.0,
            help="Threshold for buffering the selected location"
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
            
            # Display recommendations
            st.write("### Recommendations")
            for rec in data['insights']['recommendations']:
                st.write(f"- {rec}")
            
            # Display trends
            st.write("### Trend Analysis")
            trends_data = analysis['data'].get('trends', {})
            if trends_data: # Check if trends_data is not None and not empty
                for period, trend in trends_data.items():
                    if isinstance(trend, dict):
                        st.write(f"**{period.replace('_', ' ').title()}**: {trend.get('direction', 'N/A')} "
                                f"(value: {trend.get('value', 'N/A'):.3f})")
            else:
                st.write("No trend data available.")

        else:
            st.error(f"Analysis failed: {analysis.get('error', 'Unknown error')}")

def main():
    """Main entry point for the restaurant dashboard application."""
    st.set_page_config(
        page_title="BiteBase Restaurant Analytics",
        page_icon="🍽️",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    
    st.title("BiteBase Restaurant Analytics Dashboard")
    st.write("Interactive analytics for restaurant location analysis and business intelligence.")
    
    # Initialize dashboard UI
    dashboard = DashboardUI()
    
    # Show sidebar controls
    analysis_params = dashboard.show_analysis_sidebar()
    
    # Load restaurant data
    try:
        restaurant_data = RestaurantData.load_data()
        st.success(f"Loaded {len(restaurant_data)} restaurants")
    except Exception as e:
        st.error(f"Failed to load restaurant data: {str(e)}")
        restaurant_data = pd.DataFrame()  # Create empty DataFrame on error
    
    # Create tabs for different views
    tab1, tab2, tab3, tab4 = st.tabs(["Products", "Place", "Price", "Promotion"])
    
    # Create instance of MapConfig
    map_config = MapConfig()
    
    with tab1:
        st.header("Restaurant Map")
        
        # Default location (Bangkok) - Fix the attribute access
        default_lat, default_lng = MapConfig.get_default_center()  # Call the static method directly
        
        # Create map
        m = dashboard.radar_map.create_map(default_lat, default_lng)
        
        if not restaurant_data.empty:
            # Add restaurants to map
            m = dashboard.radar_map.add_restaurants_to_map(m, restaurant_data)
            
            # Add heatmap layer
            m = dashboard.radar_map.add_heatmap(m, restaurant_data)
        
        # Display the map
        st.write("Click on the map to analyze a location")
        folium_static(m)

        # Pass relevant data from the CSV to the location analysis
        sample_restaurant = restaurant_data.iloc[0].to_dict() if not restaurant_data.empty else {}
        dashboard.show_location_analysis(default_lat, default_lng, sample_restaurant)
        
        st.header("Product (Menu & Sales Insights)")
        
        # Top-Selling & Low-Performing Dishes
        st.subheader("Top-Selling & Low-Performing Dishes")
        
        # Assuming 'categories' column represents dish types and 'rating' represents sales
        if not restaurant_data.empty:
            dish_sales = restaurant_data.groupby('categories')['Customer_Ratings'].sum().nlargest(5).reset_index()
            dish_sales.columns = ['Category', 'Sales']
            st.dataframe(dish_sales, height=300)
        else:
            st.write("No restaurant data available.")
        
        st.subheader("Food Cost vs. Profitability")
        st.write("Food cost vs. profit margin data is not available in the dataset.")
        
        # Seasonal & Trend Analysis
        st.subheader("Seasonal & Trend Analysis")
        st.write("Seasonal and trend analysis data is not available in the dataset.")
        
        # Dynamic Pricing Recommendations
        st.subheader("Dynamic Pricing Recommendations")
        st.write("Dynamic pricing recommendation data is not available in the dataset.")
        
        st.write("### AI-Driven Insights (Future Implementation)")
        st.write("This section will provide AI-driven insights for product optimization, including menu recommendations and pricing strategies.")

    with tab2:
        st.header("Place (Geographic & Customer Insights)")
        
        # Customer Density Heatmap
        st.subheader("Customer Density Heatmap")
        
        if not restaurant_data.empty:
            st.map(restaurant_data[['lat', 'lng']].dropna(), zoom=13)
            st.write("Customer density based on restaurant locations.")
        else:
            st.write("No restaurant data available.")
        
        # Competitor & Market Landscape
        st.subheader("Competitor & Market Landscape")
        st.write("Competitor and market landscape data is not available in the dataset.")
        
        # Delivery & Pickup Hotspots
        st.subheader("Delivery & Pickup Hotspots")
        st.write("Delivery and pickup hotspot data is not available in the dataset.")
        
        # Real Estate & Rental Impact
        st.subheader("Real Estate & Rental Impact")
        st.write("Real estate and rental impact data is not available in the dataset.")
        
        st.write("### AI-Driven Insights (Future Implementation)")
        st.write("This section will provide AI-driven insights for location optimization, including ideal locations based on customer density and competitor analysis.")
        
    with tab3:
        st.header("Price (Revenue & Profitability Analytics)")
        
        # Sales & Revenue Forecasting
        st.subheader("Sales & Revenue Forecasting")
        
        # Dummy data for sales forecasting
        forecast_data = pd.DataFrame({
            'Month': ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun'],
            'Sales': [10000, 11000, 12000, 13000, 12500, 14000]
        })
        st.line_chart(forecast_data.set_index('Month'), height=300)
        st.write("Sales and revenue forecast for the next months.")
        
        # Peak Days & Hours Analysis
        st.subheader("Peak Days & Hours Analysis")
        
        # Dummy data for peak hours
        peak_data = pd.DataFrame({
            'Hour': [12, 13, 14, 18, 19, 20],
            'Orders': [50, 60, 55, 70, 65, 60]
        })
        st.bar_chart(peak_data.set_index('Hour'), height=300)
        st.write("Peak hours for order volume.")
        
        # Discount & Promotion Effectiveness
        st.subheader("Discount & Promotion Effectiveness")
        
        # Dummy data for discount effectiveness
        discount_data = pd.DataFrame({
            'Discount': ['None', '10%', '20%', '30%'],
            'Sales': [10000, 11000, 11500, 11200]
        })
        st.bar_chart(discount_data.set_index('Discount'), height=300)
        st.write("Sales volume with different discount levels.")
        
        # Customer Spending Behavior & Trends
        st.subheader("Customer Spending Behavior & Trends")
        st.write("### AI-Driven Insights (Future Implementation)")
        st.write("This section will provide AI-driven insights for price optimization, including dynamic pricing recommendations and promotion planning.")
        
    with tab4:
        st.header("Promotion (Marketing & Customer Engagement)")
        
        # Customer Segmentation & Loyalty Tracking
        st.subheader("Customer Segmentation & Loyalty Tracking")
        st.write("Customer segmentation and loyalty tracking data is not available in the dataset.")
        
        # Ad Performance & ROI Analysis
        st.subheader("Ad Performance & ROI Analysis")
        st.write("Ad performance and ROI analysis data is not available in the dataset.")
        
        # AI-Driven Sentiment Analysis from Reviews
        st.subheader("AI-Driven Sentiment Analysis from Reviews")
        st.write("AI-driven sentiment analysis from reviews data is not available in the dataset.")
        
        # Marketing & Seasonal Campaign Suggestions
        st.subheader("Marketing & Seasonal Campaign Suggestions")
        st.write("Marketing and seasonal campaign suggestion data is not available in the dataset.")

if __name__ == "__main__":
    main()

