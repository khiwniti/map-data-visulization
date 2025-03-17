from pydantic import BaseModel, EmailStr
from typing import Optional, Dict, Any

class Token(BaseModel):
    access_token: str
    token_type: str

class UserBase(BaseModel):
    username: str
    email: EmailStr

class UserCreate(UserBase):
    password: str
    role: str = "customer"

class User(UserBase):
    id: int
    role: str
    is_active: bool
    
    class Config:
        orm_mode = True

class StandardResponse(BaseModel):
    data: Dict[str, Any]
    meta: Dict[str, Any] = {}

class ErrorResponse(BaseModel):
    error: Dict[str, Any]

class UserResponse(StandardResponse):
    data: User