# Testing BiteBase API with Uvicorn

This guide explains how to set up and run API tests for BiteBase using Python, pytest, and a FastAPI implementation running on Uvicorn server.

## Overview

The test suite consists of:

1. `lib/fastapi-test-server.py` - A FastAPI implementation that mirrors your API structure
2. `lib/python_api_tests.py` - Python tests using pytest to verify API functionality

These tests verify that:
- All API endpoints return the expected status codes and data
- Error handling works as expected
- API performance meets requirements

## Setup for Uvicorn Testing

### Prerequisites

- Python 3.7+ 
- pip (Python package manager)

### Installation

1. Install the required dependencies:

```bash
pip install fastapi uvicorn pytest requests
```

2. Ensure the test files are in the correct location:
   - `lib/fastapi-test-server.py` (the FastAPI server)
   - `lib/python_api_tests.py` (the test suite)

## Running the Tests

### Option 1: Automatic Server Management

The test suite can automatically start and stop the Uvicorn server:

```bash
python lib/python_api_tests.py
```

This will:
1. Start the Uvicorn server
2. Run all tests
3. Shut down the server when tests are complete

### Option 2: Manual Server Management

If you prefer to run the server separately:

1. Start the Uvicorn server:
   ```bash
   python lib/fastapi-test-server.py
   ```
   
   This will run the server at http://localhost:8000 by default.

2. In another terminal, run the tests:
   ```bash
   pytest lib/python_api_tests.py -v
   ```

## Configuration Options

You can configure the tests using environment variables:

- `API_BASE_URL`: Base URL for API requests (default: `http://localhost:8000/api`)
- `TEST_AUTH_TOKEN`: Auth token for authenticated requests (default: `test-token`)
- `TEST_PROJECT_ID`: ID of test project (default: `test-project-1`)

Example:
```bash
API_BASE_URL=http://localhost:9000/api TEST_AUTH_TOKEN=custom-token pytest lib/python_api_tests.py
```

## Test Structure

The test suite is organized into classes that test different aspects of the API:

1. **TestAuthentication**: Tests auth token validation
2. **TestProjects**: Tests project listing, creation, and retrieval
3. **TestCompetitors**: Tests competitor listing and creation
4. **TestDemographics**: Tests demographic data retrieval
5. **TestTraffic**: Tests traffic data retrieval
6. **TestMenuItems**: Tests menu item listing and creation
7. **TestInsights**: Tests insight listing and creation  
8. **TestErrorHandling**: Tests error responses
9. **TestPerformance**: Tests API response times

## How It Works

The test suite works by:

1. **Starting the Server**: Launches a Uvicorn server running a FastAPI application
2. **API Requests**: Makes HTTP requests to the server to test endpoints
3. **Validation**: Verifies responses match expected format and values
4. **Server Shutdown**: Automatically terminates the server after tests complete

## Modifying Tests for Your Environment

If you need to modify the tests for a different backend:

1. Update `fastapi-test-server.py` to match your API structure
2. Adjust test expectations in `python_api_tests.py` to match your data model
3. Update any endpoint paths if they differ from the current implementation

## Troubleshooting

### Server Not Starting

If the Uvicorn server fails to start:

- Check if port 8000 is already in use
- Verify Python dependencies are installed
- Check for syntax errors in the FastAPI server code

### Failed Tests

If tests are failing:

- Check the API endpoint paths match your implementation
- Verify the data models match your API's expected formats
- Check authentication requirements match your API

### Connection Errors

If tests can't connect to the server:

- Verify the server is running (check for Uvicorn logs)
- Check the `API_BASE_URL` is correct
- Ensure there are no network restrictions blocking connections

## Extending the Tests

To add tests for additional endpoints:

1. Add the corresponding endpoint to `fastapi-test-server.py`
2. Create a new test class or add methods to existing classes in `python_api_tests.py`
3. Follow the pattern of existing tests 