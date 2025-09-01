# models.py
from sqlalchemy import Column, String, DateTime, Boolean, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.dialects.postgresql import UUID
import uuid
from datetime import datetime
from db import Base

class User(Base):
    __tablename__ = 'users'

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)  # internal UUID
    google_id = Column(String, unique=True, nullable=True)  # store Google account ID
    email = Column(String, unique=True, nullable=False)
    name = Column(String, nullable=False)
    picture = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow)

class TestUser(Base):
    __tablename__ = 'testusers'

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)  # internal UUID
    email = Column(String, unique=True, nullable=False, index=True)  # indexed for faster lookups
    name = Column(String, nullable=True)  # optional for test users
    description = Column(Text, nullable=True)  # optional description/notes
    is_active = Column(Boolean, default=True)  # can deactivate without deleting
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)