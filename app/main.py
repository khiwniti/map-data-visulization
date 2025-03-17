from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
# Import only the modules that exist
from app.api import auth, inventory, insights, analysis, projects, customers, analytics, competitors

app = FastAPI(title="BiteBaseAI API", description="Backend API for BiteBaseAI platform")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Update with specific origins in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(auth.router, prefix="/api/auth", tags=["Authentication"])
app.include_router(inventory.router, prefix="/api/inventory", tags=["Inventory"])
app.include_router(insights.router, prefix="/api/insights", tags=["Insights"])
app.include_router(analysis.router, prefix="/api/analysis", tags=["Analysis"])
app.include_router(projects.router, prefix="/api/projects", tags=["Projects"])
app.include_router(competitors.router, prefix="/api/competitors", tags=["Competitors"])
app.include_router(customers.router, prefix="/api/customers", tags=["Customers"])
app.include_router(analytics.router, prefix="/api/analytics", tags=["Analytics"])
# Removed init_db router

@app.get("/")
async def root():
    return {"message": "Welcome to BiteBaseAI API"}