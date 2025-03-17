from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.db.database import get_db

router = APIRouter()

@router.get("/")
async def get_customers(db: Session = Depends(get_db)):
    return {"message": "Customer data endpoint"}

@router.post("/")
async def create_customer(db: Session = Depends(get_db)):
    return {"message": "Customer created successfully"}