from fastapi import FastAPI, HTTPException, Depends, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from typing import List, Dict, Any, Optional
from pydantic import BaseModel
import uvicorn
import json
from datetime import datetime
import uuid

# Initialize FastAPI app
app = FastAPI(title="BiteBase API", description="BiteBase backend API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Security
security = HTTPBearer()

# Models
class Coordinates(BaseModel):
    lat: float
    lng: float

class Project(BaseModel):
    id: Optional[str] = None
    name: str
    description: Optional[str] = None
    location: str
    address: Optional[str] = None
    cuisine: str
    coordinates: Coordinates
    radius: float
    status: Optional[str] = "planning"
    userId: Optional[str] = None
    createdAt: Optional[datetime] = None
    updatedAt: Optional[datetime] = None

class Competitor(BaseModel):
    id: Optional[str] = None
    name: str
    description: Optional[str] = None
    cuisine: str
    address: str
    coordinates: Coordinates
    rating: float
    priceRange: int
    projectId: str
    createdAt: Optional[datetime] = None
    updatedAt: Optional[datetime] = None

class AgeDistribution(BaseModel):
    under18: float
    age18to24: float
    age25to34: float
    age35to44: float
    age45to54: float
    age55to64: float
    age65plus: float

class EducationLevels(BaseModel):
    highSchool: float
    someCollege: float
    bachelors: float
    graduate: float

class DemographicData(BaseModel):
    id: Optional[str] = None
    projectId: str
    populationDensity: float
    medianIncome: float
    ageDistribution: AgeDistribution
    educationLevels: EducationLevels
    createdAt: Optional[datetime] = None
    updatedAt: Optional[datetime] = None

class PeakHours(BaseModel):
    morning: int
    afternoon: int
    evening: int
    night: int

class SeasonalVariation(BaseModel):
    spring: float
    summer: float
    fall: float
    winter: float

class TrafficData(BaseModel):
    id: Optional[str] = None
    projectId: str
    weekdayAverage: int
    weekendAverage: int
    peakHours: PeakHours
    seasonalVariation: SeasonalVariation
    createdAt: Optional[datetime] = None
    updatedAt: Optional[datetime] = None

class MenuItem(BaseModel):
    id: Optional[str] = None
    name: str
    description: str
    category: str
    price: float
    cost: float
    popularity: int
    projectId: str
    createdAt: Optional[datetime] = None
    updatedAt: Optional[datetime] = None

class Insight(BaseModel):
    id: Optional[str] = None
    projectId: str
    title: str
    description: str
    category: str
    impact: str
    actionable: bool
    createdAt: Optional[datetime] = None
    updatedAt: Optional[datetime] = None

# Mock data
mock_projects = []
mock_competitors = []
mock_demographics = []
mock_traffic = []
mock_menu_items = []
mock_insights = []

# Authentication middleware
async def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    if not credentials.credentials == "test-token":
        raise HTTPException(status_code=401, detail="Invalid token")
    return {"user_id": "test-user-123"}

# Auth routes
@app.post("/api/auth/validate")
async def validate_auth(user = Depends(verify_token)):
    return {"authenticated": True}

# Project routes
@app.get("/api/projects")
async def get_projects(user = Depends(verify_token)):
    return mock_projects

@app.post("/api/projects", status_code=201)
async def create_project(project: Project, user = Depends(verify_token)):
    new_project = project.dict()
    new_project["id"] = str(uuid.uuid4())
    new_project["userId"] = user["user_id"]
    new_project["createdAt"] = datetime.now()
    new_project["updatedAt"] = datetime.now()
    mock_projects.append(new_project)
    return new_project

@app.get("/api/projects/{project_id}")
async def get_project(project_id: str, user = Depends(verify_token)):
    for project in mock_projects:
        if project["id"] == project_id:
            return project
    raise HTTPException(status_code=404, detail="Project not found")

# Competitors API
@app.get("/api/inventory")
async def get_competitors(projectId: str, user = Depends(verify_token)):
    return [comp for comp in mock_competitors if comp["projectId"] == projectId]

@app.post("/api/inventory", status_code=201)
async def create_competitor(competitor: Competitor, user = Depends(verify_token)):
    new_competitor = competitor.dict()
    new_competitor["id"] = str(uuid.uuid4())
    new_competitor["createdAt"] = datetime.now()
    new_competitor["updatedAt"] = datetime.now()
    mock_competitors.append(new_competitor)
    return new_competitor

# Demographics API
@app.get("/api/analytics/demographics")
async def get_demographics(projectId: str, user = Depends(verify_token)):
    for demo in mock_demographics:
        if demo["projectId"] == projectId:
            return demo
    
    # Create mock data if none exists
    mock_demo = {
        "id": str(uuid.uuid4()),
        "projectId": projectId,
        "populationDensity": 5000,
        "medianIncome": 75000,
        "ageDistribution": {
            "under18": 15,
            "age18to24": 10,
            "age25to34": 25,
            "age35to44": 20,
            "age45to54": 15,
            "age55to64": 10,
            "age65plus": 5
        },
        "educationLevels": {
            "highSchool": 25,
            "someCollege": 30,
            "bachelors": 35,
            "graduate": 10
        },
        "createdAt": datetime.now(),
        "updatedAt": datetime.now()
    }
    mock_demographics.append(mock_demo)
    return mock_demo

# Traffic API
@app.get("/api/analytics/traffic")
async def get_traffic(projectId: str, user = Depends(verify_token)):
    for traffic in mock_traffic:
        if traffic["projectId"] == projectId:
            return traffic
    
    # Create mock data if none exists
    mock_traffic_data = {
        "id": str(uuid.uuid4()),
        "projectId": projectId,
        "weekdayAverage": 1000,
        "weekendAverage": 1500,
        "peakHours": {
            "morning": 300,
            "afternoon": 500,
            "evening": 700,
            "night": 200
        },
        "seasonalVariation": {
            "spring": 1.1,
            "summer": 1.3,
            "fall": 1.0,
            "winter": 0.8
        },
        "createdAt": datetime.now(),
        "updatedAt": datetime.now()
    }
    mock_traffic.append(mock_traffic_data)
    return mock_traffic_data

# Menu Items API
@app.get("/api/analytics/menu-items")
async def get_menu_items(projectId: str, user = Depends(verify_token)):
    items = [item for item in mock_menu_items if item["projectId"] == projectId]
    if not items:
        # Create a mock item if none exists
        mock_item = {
            "id": str(uuid.uuid4()),
            "name": "Test Menu Item",
            "description": "Description of test menu item",
            "category": "Appetizer",
            "price": 9.99,
            "cost": 3.50,
            "popularity": 75,
            "projectId": projectId,
            "createdAt": datetime.now(),
            "updatedAt": datetime.now()
        }
        mock_menu_items.append(mock_item)
        return [mock_item]
    return items

@app.post("/api/analytics/menu-items", status_code=201)
async def create_menu_item(menu_item: MenuItem, user = Depends(verify_token)):
    new_item = menu_item.dict()
    new_item["id"] = str(uuid.uuid4())
    new_item["createdAt"] = datetime.now()
    new_item["updatedAt"] = datetime.now()
    mock_menu_items.append(new_item)
    return new_item

# Insights API
@app.get("/api/insights")
async def get_insights(projectId: str, user = Depends(verify_token)):
    insights = [insight for insight in mock_insights if insight["projectId"] == projectId]
    if not insights:
        # Create a mock insight if none exists
        mock_insight = {
            "id": str(uuid.uuid4()),
            "projectId": projectId,
            "title": "Test Insight",
            "description": "Description of test insight",
            "category": "price",
            "impact": "high",
            "actionable": True,
            "createdAt": datetime.now(),
            "updatedAt": datetime.now()
        }
        mock_insights.append(mock_insight)
        return [mock_insight]
    return insights

@app.post("/api/insights", status_code=201)
async def create_insight(insight: Insight, user = Depends(verify_token)):
    new_insight = insight.dict()
    new_insight["id"] = str(uuid.uuid4())
    new_insight["createdAt"] = datetime.now()
    new_insight["updatedAt"] = datetime.now()
    mock_insights.append(new_insight)
    return new_insight

# Error handling for test cases
@app.post("/api/projects/error-test")
async def project_error_test(request: Request):
    body = await request.json()
    if "name" not in body or "location" not in body or "cuisine" not in body:
        raise HTTPException(status_code=400, detail="Name, location, and cuisine are required")
    return {"success": True}

# Create some initial test data
def initialize_mock_data():
    # Add a test project
    if not mock_projects:
        test_project = {
            "id": "test-project-1",
            "name": "Test Restaurant",
            "description": "Test project for API testing",
            "location": "Test Location",
            "address": "123 Test Street",
            "cuisine": "Test Cuisine",
            "coordinates": {"lat": 40.123, "lng": -74.456},
            "radius": 1.5,
            "status": "active",
            "userId": "test-user-123",
            "createdAt": datetime.now(),
            "updatedAt": datetime.now()
        }
        mock_projects.append(test_project)

if __name__ == "__main__":
    initialize_mock_data()
    uvicorn.run(app, host="127.0.0.1", port=8001)  # Changed port from 8000 to 8001
