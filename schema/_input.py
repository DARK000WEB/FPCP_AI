from pydantic import BaseModel, Field, EmailStr
from typing import Optional


class UserInput(BaseModel):
    username: str = Field(..., max_length=50)
    phone_number: str = Field(..., pattern=r"^\d{11}$")
    degree: str
    password: str = Field(..., min_length=8, max_length=40)
    confirm_password: str = Field(..., min_length=8, max_length=40)


class UserInputLogin(BaseModel):
    username: str = Field(..., max_length=50)
    password: str = Field(..., min_length=8, max_length=40)


class UpdateUserProfileInput(BaseModel):
    username: Optional[str] = Field(None, max_length=50)
    family: Optional[str] = Field(None, max_length=40)
    age: Optional[int] = Field(None, ge=0)
    phone_number: str = Field(..., pattern=r"^\d{11}$")
    degree: Optional[str] = Field(None, max_length=40)
    email: Optional[EmailStr] = None


class TextCreateRequest(BaseModel):
    text_content: str
