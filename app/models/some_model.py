# Change this:
class SomeModel(BaseModel):
    class Config:
        orm_mode = True

# To this:
class SomeModel(BaseModel):
    class Config:
        from_attributes = True