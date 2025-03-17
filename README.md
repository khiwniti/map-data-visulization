# BiteBaseAI Platform

## Overview
BiteBaseAI is a comprehensive data analytics platform for restaurant and food service businesses, providing insights, inventory management, competitor analysis, and customer analytics.

## Components

### Authentication
Handles user management, authentication, and session tracking.

### Inventory
Manages inventory data and related insights.

### Analysis
Provides analytical capabilities including user retention metrics, conversion rates, and engagement analysis.

### Database Initialization
Handles database schema migrations and maintenance.

## Data Dictionary
For a complete data dictionary and mapping information, see [data_mapping.md](data_mapping.md).

## API Documentation

### Authentication
- **POST /api/auth/login**: Authenticates users and returns access token
- **POST /api/auth/register**: Registers new users

### Inventory
- **GET /api/inventory/items**: Retrieves inventory items
- **POST /api/inventory/items**: Creates new inventory items
- **POST /api/inventory/suppliers**: Creates new suppliers

### Projects
- **GET /api/projects/**: Lists all projects for the current user
- **POST /api/projects/**: Creates a new project
- **GET /api/projects/{id}**: Gets a specific project
- **PUT /api/projects/{id}**: Updates a project
- **DELETE /api/projects/{id}**: Deletes a project

### Competitors
- **GET /api/projects/{id}/competitors**: Lists competitors for a project
- **POST /api/projects/{id}/competitors**: Adds a competitor
- **GET /api/projects/{id}/competitors/{competitorId}**: Gets a specific competitor
- **PUT /api/projects/{id}/competitors/{competitorId}**: Updates a competitor
- **DELETE /api/projects/{id}/competitors/{competitorId}**: Deletes a competitor

### Analysis
- **POST /api/analysis**: Initiates analysis for a project

### Insights
- **GET /api/projects/{id}/insights**: Gets insights for a project
- **GET /api/projects/{id}/insights/{insightId}**: Gets a specific insight

### Staff
- **GET /api/staff**: Gets staff information
- **POST /api/staff**: Creates new staff records

### Customers
- **GET /api/customers**: Gets customer information
- **POST /api/customers**: Creates new customer records

### Analytics
- **GET /api/analytics/dashboard**: Gets dashboard analytics
- **GET /api/analytics/reports**: Gets analytical reports

## Getting Started

### Installation
```bash
# Clone the repository
git clone <repository-url>

# Set up virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Start the application
uvicorn app.main:app --reload
```

### Environment Variables
Create a `.env` file in the root directory with the following variables:
```
DATABASE_URL=postgresql://user:password@localhost/dbname
SECRET_KEY=your_secret_key
```

## Development

### Database Migrations
```bash
# Initialize migrations
alembic init migrations

# Create a new migration
alembic revision --autogenerate -m "Description"

# Apply migrations
alembic upgrade head
```

## Contributing
[Add contribution guidelines here]

## License
[Add license information here]