# BiteBase Restaurant Analytics Dashboard

A comprehensive analytics dashboard for restaurant location analysis and business intelligence.

## Features

- Interactive map visualization with Kepler.gl
- Real-time location change predictions
- Business analytics with PyGWalker
- Historical trend analysis
- Dynamic risk assessment
- Multi-factor location scoring

## Project Structure

```
.
├── RAW/                      # Raw data files
│   ├── Dynamics/            # Dynamic data sources
│   │   └── LMWN/           # Restaurant data
│   └── Statics/            # Static reference data
├── src/                     # Source code
│   ├── data/               # Data processing modules
│   ├── models/             # ML models and predictors
│   │   ├── combined_location_model.py
│   │   ├── location_change_prediction_model.py
│   │   ├── realtime_location_change_model.py
│   │   └── saved_model.pkl
│   ├── utils/              # Utility functions
│   │   └── mock_data_generator.py
│   └── restaurant_dashboard.py  # Main dashboard application
├── requirements.txt         # Project dependencies
└── README.md               # Project documentation
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/map-data-visulization.git
cd map-data-visulization
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Run the dashboard:
```bash
streamlit run src/restaurant_dashboard.py
```

2. Use the interactive map:
- Click on any location to get real-time analysis
- Use sidebar filters to customize the view
- Explore different analytics tabs for detailed insights

3. View analytics:
- Product Analytics: Menu performance and sales trends
- Place Analytics: Geographic and competitive analysis
- Price Analytics: Revenue and profitability metrics
- Promotion Analytics: Marketing and customer engagement

## Models

### Combined Location Model
- Integrates static and real-time predictions
- Uses historical data for trend analysis
- Provides explainable AI insights

### Real-time Location Change Model
- Dynamic risk assessment
- Time-sensitive predictions
- Trend monitoring and alerts

## Data Sources

- Restaurant information from LMWN API
- Geographic data for location analysis
- Historical performance metrics
- Real-time business indicators

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Streamlit for the web framework
- Kepler.gl for map visualization
- PyGWalker for analytics tools