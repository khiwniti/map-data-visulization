from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from typing import List
from app.db.database import get_db
from app.db.models.inventory import InventoryItem, Supplier
from app.schemas.inventory import InventoryItemCreate, InventoryItemResponse, SupplierCreate, SupplierResponse
from app.core.security import get_current_user
from app.db.models.users import User

router = APIRouter()

@router.get("/items", response_model=List[InventoryItemResponse])
async def get_inventory_items(
    skip: int = 0, 
    limit: int = 100, 
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    items = db.query(InventoryItem).offset(skip).limit(limit).all()
    return [{"data": item, "meta": {}} for item in items]

@router.post("/items", response_model=InventoryItemResponse)
async def create_inventory_item(
    item: InventoryItemCreate, 
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    if current_user.role not in ["admin", "staff"]:
        raise HTTPException(status_code=403, detail="Not authorized to create inventory items")
    
    db_item = InventoryItem(**item.dict())
    db.add(db_item)
    db.commit()
    db.refresh(db_item)
    
    return {"data": db_item, "meta": {}}

@router.get("/suppliers", response_model=List[SupplierResponse])
async def get_suppliers(
    skip: int = 0, 
    limit: int = 100, 
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    suppliers = db.query(Supplier).offset(skip).limit(limit).all()
    return [{"data": supplier, "meta": {}} for supplier in suppliers]

@router.post("/suppliers", response_model=SupplierResponse)
async def create_supplier(
    supplier: SupplierCreate, 
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    if current_user.role != "admin":
        raise HTTPException(status_code=403, detail="Not authorized to create suppliers")
    
    db_supplier = Supplier(**supplier.dict())
    db.add(db_supplier)
    db.commit()
    db.refresh(db_supplier)
    
    return {"data": db_supplier, "meta": {}}