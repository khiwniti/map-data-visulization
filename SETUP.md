# BiteBase Server Setup Instructions

## Prerequisites
- Python 3.8 or higher
- pip (Python package installer)
- Git

## Installation Steps

1. Clone the repository:
```bash
git clone https://github.com/yourusername/map-data-visulization.git
cd map-data-visulization
```

2. Create and activate a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install required packages:
```bash
pip install -r requirements.txt
```

## Testing the Server

1. Run the test server:
```bash
streamlit run test_server.py
```
You should see a test dashboard with sample data and controls.

2. If the test is successful, run the main dashboard:
```bash
streamlit run src/restaurant_dashboard.py
```

## Troubleshooting

### Common Issues

1. Import errors:
- Make sure you're in the project root directory
- Verify that all required packages are installed
- Check that Python path includes the project directory

2. Server not starting:
- Check if port 8501 is available
- Verify Streamlit installation
- Try running with different port:
```bash
streamlit run src/restaurant_dashboard.py --server.port 8502
```

3. Data loading issues:
- Verify RAW data directory structure
- Check file permissions
- Ensure all data files are present

### Quick Test Commands

Test the basic setup:
```bash
python src/main.py
```

Test data generation:
```bash
python src/utils/mock_data_generator.py
```

### Getting Help

If you encounter issues:
1. Check the logs in .streamlit/logs
2. Verify environment variables
3. Ensure all paths in config files are correct
4. Contact support with error messages and logs

## Development Setup

For development work:
```bash
pip install -r requirements-dev.txt  # Install development dependencies
python -m pytest tests/  # Run tests
```

## Configuration

Key configuration files:
- `src/data/__init__.py`: Data paths
- `src/models/__init__.py`: Model configuration
- `.streamlit/config.toml`: Streamlit settings

## Usage

1. Start the dashboard:
```bash
streamlit run src/restaurant_dashboard.py
```

2. Open in browser:
- Local: http://localhost:8501
- Network: http://YOUR_IP:8501

3. Use the sidebar controls to:
- Filter data
- Adjust parameters
- View different insights

4. Interact with the map to:
- Select locations
- View predictions
- Analyze trends