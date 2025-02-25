import streamlit as st
import folium
from streamlit_folium import st_folium
import json
import pandas as pd
from geopy.distance import geodesic
from pathlib import Path

# Page config
st.set_page_config(page_title="Restaurant Dashboard", layout="wide")

# Constants
DEFAULT_CENTER = [13.7563, 100.5018]  # Bangkok
DEFAULT_ZOOM = 13
DEFAULT_RADIUS = 5  # km

def load_restaurant_data():
    """Load and process restaurant data from JSON file."""
    data_path = Path("RAW/Dynamics/LMWN/restaurants.json")
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    restaurants = []
    for restaurant in data['page']['entities']:
        try:
            if restaurant.get('lat') and restaurant.get('lng'):
                categories = []
                if restaurant.get('categories'):
                    for cat in restaurant['categories']:
                        # Use both Thai name and international name if available
                        thai_name = cat.get('name', '')
                        int_name = cat.get('internationalName', '')
                        if int_name:
                            categories.append(f"{thai_name} ({int_name})")
                        else:
                            categories.append(thai_name)

                restaurants.append({
                    'name': restaurant.get('name', ''),
                    'lat': restaurant['lat'],
                    'lng': restaurant['lng'],
                    'categories': categories,
                    'address': restaurant.get('contact', {}).get('address', {}).get('street', ''),
                    'phone': restaurant.get('contact', {}).get('phoneno', ''),
                    'price_range': restaurant.get('priceRange', {}).get('name', 'N/A')
                })
        except Exception as e:
            st.error(f"Error processing restaurant: {e}")
            continue
            
    return pd.DataFrame(restaurants)

def create_map(center, zoom_start=13):
    """Create a folium map instance."""
    m = folium.Map(
        location=center,
        zoom_start=zoom_start,
        tiles="OpenStreetMap"
    )
    return m

def add_restaurant_markers(m, df, target_location=None, selected_categories=None, max_distance=None):
    """Add restaurant markers to the map."""
    for idx, row in df.iterrows():
        # Apply category filter
        if selected_categories:
            matches = False
            for rest_cat in row['categories']:
                for sel_cat in selected_categories:
                    if sel_cat in rest_cat:  # Check if selected category is part of restaurant category
                        matches = True
                        break
                if matches:
                    break
            if not matches:
                continue

        # Calculate distance if target location is set
        distance = None
        if target_location:
            distance = geodesic(target_location, (row['lat'], row['lng'])).kilometers
            if max_distance and distance > max_distance:
                continue

        # Create popup content
        popup_html = f"""
        <div style='width: 200px'>
            <h4>{row['name']}</h4>
            <p><b>Categories:</b> {', '.join(row['categories'])}</p>
            <p><b>Price Range:</b> {row['price_range']}</p>
            {"<p><b>Distance:</b> {:.2f} km</p>".format(distance) if distance else ""}
            <p><b>Address:</b> {row['address']}</p>
            <p><b>Phone:</b> {row['phone']}</p>
        </div>
        """
        
        # Add marker
        folium.Marker(
            location=[row['lat'], row['lng']],
            popup=folium.Popup(popup_html, max_width=300),
            icon=folium.Icon(color='red', icon='info-sign')
        ).add_to(m)

def main():
    st.title("Restaurant Map Dashboard")

    # Load data
    with st.spinner("Loading restaurant data..."):
        df = load_restaurant_data()

    # Sidebar filters
    st.sidebar.header("Filters")

    # Insight type filter
    insight_type = st.sidebar.selectbox(
        "Select Insight Type",
        options=["Product", "Place", "Price", "Promotion"]
    )

    if insight_type == "Product":
        st.sidebar.subheader("Product Insights")
        # Add filters and visualizations for Product insights
        # Example: Top-Selling & Low-Performing Dishes
        # ...additional code for Product insights...

    elif insight_type == "Place":
        st.sidebar.subheader("Place Insights")
        # Add filters and visualizations for Place insights
        # Example: Customer Density Heatmap
        # ...additional code for Place insights...

    elif insight_type == "Price":
        st.sidebar.subheader("Price Insights")
        # Add filters and visualizations for Price insights
        # Example: Sales & Revenue Forecasting
        # ...additional code for Price insights...

    elif insight_type == "Promotion":
        st.sidebar.subheader("Promotion Insights")
        # Add filters and visualizations for Promotion insights
        # Example: Customer Segmentation & Loyalty Tracking
        # ...additional code for Promotion insights...

    # Category filter
    all_categories = set()
    for cats in df['categories']:
        all_categories.update(cats)
    selected_categories = st.sidebar.multiselect(
        "Select Categories",
        options=sorted(list(all_categories)),
        default=None
    )

    # Distance filter
    max_distance = st.sidebar.slider(
        "Maximum Distance (km)",
        min_value=1,
        max_value=20,
        value=DEFAULT_RADIUS
    )

    # Create base map
    m = create_map(DEFAULT_CENTER)

    # Add target pin functionality
    st.write("Click on the map to set a target location")
    map_data = st_folium(m, width=800, height=600)

    target_location = None
    if map_data['last_clicked']:
        target_location = [
            map_data['last_clicked']['lat'],
            map_data['last_clicked']['lng']
        ]
        
        # Add target marker
        m = create_map(target_location)
        folium.Marker(
            location=target_location,
            popup="Target Location",
            icon=folium.Icon(color='green', icon='flag')
        ).add_to(m)
        
        # Add restaurant markers with distance calculation
        add_restaurant_markers(m, df, target_location, selected_categories, max_distance)
        
        # Display updated map
        st_folium(m, width=800, height=600)
        
        # Show statistics
        if target_location:
            restaurants_in_range = sum(1 for idx, row in df.iterrows() 
                if geodesic(target_location, (row['lat'], row['lng'])).kilometers <= max_distance)
            st.write(f"Restaurants within {max_distance}km: {restaurants_in_range}")
    else:
        # Show all restaurants without distance filtering
        add_restaurant_markers(m, df, selected_categories=selected_categories)
        st_folium(m, width=800, height=600)

if __name__ == "__main__":
    main()
