from uuid import UUID, uuid4
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy import ForeignKey, String
from datetime import datetime
from sqlalchemy.ext.declarative import declarative_base
from .engine import Base


# User Model
class UserModel(Base):
    __tablename__ = "users"

    username: Mapped[str] = mapped_column(nullable=False, unique=True)
    email: Mapped[str] = mapped_column(nullable=True, unique=True)
    password: Mapped[str] = mapped_column(nullable=False)
    age: Mapped[int] = mapped_column(nullable=True)
    phone_number: Mapped[str] = mapped_column(nullable=False)
    family: Mapped[str] = mapped_column(nullable=True)
    degree: Mapped[str] = mapped_column(nullable=False)
    profile_image: Mapped[str] = mapped_column(String, nullable=True)
    recovery_code: Mapped[str] = mapped_column(String(length=100), nullable=True)
    id: Mapped[UUID] = mapped_column(primary_key=True, default_factory=uuid4)


# Text Model
class TextModel(Base):
    __tablename__ = "texts"

    text: Mapped[str] = mapped_column(nullable=False)
    user_id: Mapped[UUID] = mapped_column(ForeignKey("users.id"))
    creation_date: Mapped[datetime] = mapped_column(default=datetime.utcnow)
    id: Mapped[UUID] = mapped_column(primary_key=True, default_factory=uuid4)


# Article Model
class ArticleModel(Base):
    __tablename__ = "articles"

    title: Mapped[str] = mapped_column(nullable=False)
    header: Mapped[str] = mapped_column(nullable=False)
    content: Mapped[str] = mapped_column(nullable=False)
    link: Mapped[str] = mapped_column(nullable=True)
    snippet: Mapped[str] = mapped_column(nullable=True)
    user_id: Mapped[UUID] = mapped_column(ForeignKey("users.id"))
    text_id: Mapped[UUID] = mapped_column(ForeignKey("texts.id"))
    created_at: Mapped[datetime] = mapped_column(default=datetime.utcnow)
    id: Mapped[UUID] = mapped_column(primary_key=True, default_factory=uuid4)


# AI Model
class AIModel(Base):
    __tablename__ = "ai_models"

    title: Mapped[str] = mapped_column(nullable=False)
    article_id: Mapped[UUID] = mapped_column(ForeignKey("articles.id"), nullable=False)
    text_id: Mapped[UUID] = mapped_column(ForeignKey("texts.id"), nullable=False)
    user_id: Mapped[UUID] = mapped_column(ForeignKey("users.id"), nullable=False)
    model_output: Mapped[str] = mapped_column(nullable=False)
    translated_output: Mapped[str] = mapped_column(nullable=True)
    created_at: Mapped[datetime] = mapped_column(default=datetime.utcnow)
    id: Mapped[UUID] = mapped_column(primary_key=True, default_factory=uuid4)


# File Model
class FileModel(Base):
    __tablename__ = "files"

    ai_model_id: Mapped[UUID] = mapped_column(
        ForeignKey("ai_models.id"), nullable=False
    )
    user_id: Mapped[UUID] = mapped_column(ForeignKey("users.id"), nullable=False)
    file_name: Mapped[str] = mapped_column(nullable=False)
    file_type: Mapped[str] = mapped_column(nullable=False)
    file_path: Mapped[str] = mapped_column(nullable=False)
    file_type: Mapped[str] = mapped_column(nullable=False)
    created_at: Mapped[datetime] = mapped_column(default=datetime.utcnow)
    id: Mapped[UUID] = mapped_column(primary_key=True, default_factory=uuid4)
