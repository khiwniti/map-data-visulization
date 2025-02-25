import streamlit as st
import folium
import plotly.express as px
import pandas as pd
import numpy as np
from folium.plugins import HeatMap
import random
import json
from location_change_prediction import train_location_change_model, predict_location_change

def load_data():
    # Load data from JSON file
    with open('RAW/Dynamics/LMWN/restaurants.json', 'r') as f:
        json_data = json.load(f)
    df = parse_json(json_data)
    return df

def parse_json(json_data):
    parsed_data = []
    for entity in json_data['page']['entities']:
        parsed_data.append({
            'name': entity['displayName'],
            'branch': entity['branch']['primary'] if entity.get('branch') else None,
            'lat': entity['lat'],
            'lng': entity['lng'],
            'category': entity['categories'][0]['name'] if entity.get('categories') else None,
            'price_range': entity['priceRange']['name'] if entity.get('priceRange') else None,
            'rating': entity['rating'],
            'reviews': entity['statistic']['numberOfReviews'],
            'contact': entity['contact']['phoneno'] if entity.get('contact') else None,
            'email': entity['contact']['email'] if entity.get('contact') else None,
            'menu_url': entity['menu']['texts']['url'] if entity.get('menu') else None,
            'photo_url': entity['defaultPhoto']['contentUrl'] if entity.get('defaultPhoto') else None,
            'location_change': random.choice([0, 1])  # Placeholder for location change (0 or 1)
        })
    return pd.DataFrame(parsed_data)

def create_map(data):
    from folium.plugins import HeatMapWithTime

    # Create a synthetic time component
    num_time_steps = 10
    data['time'] = np.repeat(np.arange(num_time_steps), len(data) // num_time_steps)

    # Create a list of lists, where each inner list contains the lat/lng data for a specific time step
    locations = []
    for t in range(num_time_steps):
        time_data = data[data['time'] == t]
        locations.append(time_data[['lat', 'lng']].values.tolist())

    # Create a Folium map centered on the mean coordinates
    m = folium.Map(location=[data['lat'].mean(), data['lng'].mean()], zoom_start=12)

    # Add HeatMapWithTime
    HeatMapWithTime(locations, radius=15, auto_play=True, max_opacity=0.8).add_to(m)

    return m._repr_html_()

def generate_charts(data):
    fig = px.scatter(data, x='price_range', y='rating', size='reviews', color='category',
                     title='Price vs Rating Analysis')
    return fig

def update_dashboard():
    data = load_data()
    map_html = create_map(data)
    chart = generate_charts(data)
    return map_html, chart

def top_selling_low_performing_dishes(data):
    # Placeholder for top-selling and low-performing dishes analysis
    return "Top-Selling & Low-Performing Dishes Analysis"

def food_cost_vs_profitability(data):
    # Placeholder for food cost vs. profitability analysis
    return "Food Cost vs. Profitability Analysis"

def seasonal_trend_analysis(data):
    # Placeholder for seasonal and trend analysis
    return "Seasonal & Trend Analysis"

def dynamic_pricing_recommendations(data):
    # Placeholder for dynamic pricing recommendations
    return "Dynamic Pricing Recommendations"

def competitor_market_landscape(data):
    # Placeholder for competitor and market landscape analysis
    return "Competitor & Market Landscape Analysis"

def delivery_pickup_hotspots(data):
    # Placeholder for delivery and pickup hotspots analysis
    return "Delivery & Pickup Hotspots Analysis"

def real_estate_rental_impact(data):
    # Placeholder for real estate and rental impact analysis
    return "Real Estate & Rental Impact Analysis"

def sales_revenue_forecasting(data):
    # Placeholder for sales and revenue forecasting
    return "Sales & Revenue Forecasting"

def peak_days_hours_analysis(data):
    # Placeholder for peak days and hours analysis
    return "Peak Days & Hours Analysis"

def discount_promotion_effectiveness(data):
    # Placeholder for discount and promotion effectiveness analysis
    return "Discount & Promotion Effectiveness Analysis"

def customer_spending_behavior_trends(data):
    # Placeholder for customer spending behavior and trends analysis
    return "Customer Spending Behavior & Trends Analysis"

def customer_segmentation_loyalty_tracking(data):
    # Placeholder for customer segmentation and loyalty tracking
    return "Customer Segmentation & Loyalty Tracking"

def ad_performance_roi_analysis(data):
    # Placeholder for ad performance and ROI analysis
    return "Ad Performance & ROI Analysis"

def ai_driven_sentiment_analysis(data):
    # Placeholder for AI-driven sentiment analysis from reviews
    return "AI-Driven Sentiment Analysis from Reviews"

def marketing_seasonal_campaign_suggestions(data):
    # Placeholder for marketing and seasonal campaign suggestions
    return "Marketing & Seasonal Campaign Suggestions"

def real_time_sales_inventory_tracking(data):
    # Placeholder for real-time sales and inventory tracking
    return "Real-Time Sales & Inventory Tracking"

def smart_labor_scheduling_staff_efficiency(data):
    # Placeholder for smart labor scheduling and staff efficiency
    return "Smart Labor Scheduling & Staff Efficiency"

def multi_branch_performance_dashboard(data):
    # Placeholder for multi-branch performance dashboard
    return "Multi-Branch Performance Dashboard"

def financial_traffic_forecasts_expansion_planning(data):
    # Placeholder for financial and traffic forecasts for expansion planning
    return "Financial & Traffic Forecasts for Expansion Planning"

st.title("🏢 BiteBase Restaurant Dashboard")

def display_dashboard_elements(update_dashboard):
    map_output = st.empty()
    analytics_chart = st.empty()
    prediction_output = st.empty()

    map_html, chart = update_dashboard()
    st.components.v1.html(map_html, height=600)
    analytics_chart.plotly_chart(chart)
    return prediction_output

prediction_output = display_dashboard_elements(update_dashboard)

data = load_data()

def train_and_predict():
    data = load_data()
    model = train_location_change_model(data)
    if model:
        prediction_data = predict_location_change(model, data)
        return prediction_data
    else:
        return "Model training failed."

if st.button("Predict Location Change"):
    prediction_data = train_and_predict()
    if isinstance(prediction_data, str):
        prediction_output.write(prediction_data)
    else:
        st.dataframe(prediction_data[['name', 'location_change_prediction']])

# Display additional insights
st.subheader("Product (Menu & Sales Insights)")
st.write(top_selling_low_performing_dishes(data))
st.write(food_cost_vs_profitability(data))
st.write(seasonal_trend_analysis(data))
st.write(dynamic_pricing_recommendations(data))

st.subheader("Place (Geographic & Customer Insights)")
st.write(competitor_market_landscape(data))
st.write(delivery_pickup_hotspots(data))
st.write(real_estate_rental_impact(data))

st.subheader("Price (Revenue & Profitability Analytics)")
st.write(sales_revenue_forecasting(data))
st.write(peak_days_hours_analysis(data))
st.write(discount_promotion_effectiveness(data))
st.write(customer_spending_behavior_trends(data))

st.subheader("Promotion (Marketing & Customer Engagement)")
st.write(customer_segmentation_loyalty_tracking(data))
st.write(ad_performance_roi_analysis(data))
st.write(ai_driven_sentiment_analysis(data))
st.write(marketing_seasonal_campaign_suggestions(data))
st.write(real_time_sales_inventory_tracking(data))
st.write(smart_labor_scheduling_staff_efficiency(data))
st.write(multi_branch_performance_dashboard(data))
st.write(financial_traffic_forecasts_expansion_planning(data))
