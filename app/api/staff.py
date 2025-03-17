from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from typing import List, Optional
from app.db.database import get_db

router = APIRouter()

@router.get("/")
async def get_staff():
    """
    Get staff information
    """
    return {"message": "Staff API endpoint"}