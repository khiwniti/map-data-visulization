from sqlalchemy import Column, DateTime, Integer, String, Text, Numeric, ForeignKey, func, Date
from app.db.database import Base

class Supplier(Base):
    __tablename__ = "suppliers"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(100), nullable=False)
    contact_name = Column(String(100))
    email = Column(String(100))
    phone = Column(String(20))
    address = Column(Text)

class InventoryItem(Base):
    __tablename__ = "inventory_items"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(100), nullable=False)
    description = Column(Text)
    category = Column(String(50))
    sku = Column(String(50), unique=True)
    barcode = Column(String(50))
    unit_price = Column(Numeric(10, 2), nullable=False)
    cost_price = Column(Numeric(10, 2), nullable=False)
    quantity = Column(Integer, nullable=False, default=0)
    reorder_level = Column(Integer)
    supplier_id = Column(Integer, ForeignKey("suppliers.id"))
    storage_location = Column(String(100))
    expiry_date = Column(Date)
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())