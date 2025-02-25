# BiteBase Restaurant Dashboard

## Overview

The BiteBase Restaurant Dashboard is a Streamlit application that provides various insights and analytics for restaurant data. It includes features such as geographic and customer insights, revenue and profitability analytics, and marketing and customer engagement insights. The dashboard also includes a machine learning model to predict location changes for restaurants.

## Features

1. **Product (Menu & Sales Insights)**
   - Top-Selling & Low-Performing Dishes
   - Food Cost vs. Profitability
   - Seasonal & Trend Analysis
   - Dynamic Pricing Recommendations

2. **Place (Geographic & Customer Insights)**
   - Customer Density Heatmap
   - Competitor & Market Landscape
   - Delivery & Pickup Hotspots
   - Real Estate & Rental Impact

3. **Price (Revenue & Profitability Analytics)**
   - Sales & Revenue Forecasting
   - Peak Days & Hours Analysis
   - Discount & Promotion Effectiveness
   - Customer Spending Behavior & Trends

4. **Promotion (Marketing & Customer Engagement)**
   - Customer Segmentation & Loyalty Tracking
   - Ad Performance & ROI Analysis
   - AI-Driven Sentiment Analysis from Reviews
   - Marketing & Seasonal Campaign Suggestions
   - Real-Time Sales & Inventory Tracking
   - Smart Labor Scheduling & Staff Efficiency
   - Multi-Branch Performance Dashboard
   - Financial & Traffic Forecasts for Expansion Planning

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/map-data-visualization.git
    cd map-data-visualization
    ```

2. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

3. Ensure you have the necessary data files in the `RAW/Dynamics/LMWN/` directory.

## Usage

1. Run the Streamlit application:
    ```bash
    streamlit run location_change_prediction.py
    ```

2. Open your web browser and navigate to the provided URL (e.g., `http://localhost:8501`).

## File Structure

- `restaurant_dashboard.py`: Main Streamlit dashboard code.
- `location_change_prediction.py`: Functions for training the machine learning model and making predictions.
- `RAW/Dynamics/LMWN/restaurants.json`: JSON file containing restaurant data.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
