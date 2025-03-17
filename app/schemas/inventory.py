from pydantic import BaseModel
from typing import Optional, Dict, Any
from datetime import date, datetime

class SupplierBase(BaseModel):
    name: str
    contact_name: Optional[str] = None
    email: Optional[str] = None
    phone: Optional[str] = None
    address: Optional[str] = None

class SupplierCreate(SupplierBase):
    pass

class Supplier(SupplierBase):
    id: int
    
    class Config:
        orm_mode = True

class InventoryItemBase(BaseModel):
    name: str
    description: Optional[str] = None
    category: Optional[str] = None
    sku: Optional[str] = None
    barcode: Optional[str] = None
    unit_price: float
    cost_price: float
    quantity: int = 0
    reorder_level: Optional[int] = None
    supplier_id: Optional[int] = None
    storage_location: Optional[str] = None
    expiry_date: Optional[date] = None

class InventoryItemCreate(InventoryItemBase):
    pass

class InventoryItem(InventoryItemBase):
    id: int
    created_at: datetime
    updated_at: datetime
    
    class Config:
        orm_mode = True

class StandardResponse(BaseModel):
    data: Dict[str, Any]
    meta: Dict[str, Any] = {}

class InventoryItemResponse(StandardResponse):
    data: InventoryItem

class SupplierResponse(StandardResponse):
    data: Supplier