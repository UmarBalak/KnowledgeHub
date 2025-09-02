from sqlalchemy import (
    Column,
    Integer,
    String,
    DateTime,
    Text,
    Boolean,
    func,
    Enum
)
from database import Base
from datetime import datetime
import enum

class RoleEnum(str, enum.Enum):
    maintainer = "maintainer"
    user = "user"

class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    google_id = Column(String(50), unique=True, nullable=True)  # Google ID max ~21 chars, 50 is safe
    email = Column(String(255), unique=True, nullable=False, index=True)  # safe for email storage
    name = Column(String(100), nullable=False)  # reasonable length for names
    picture = Column(String(500))  # profile picture URL can be long
    role = Column(Enum(RoleEnum), nullable=False, default=RoleEnum.user.value)
    created_at = Column(DateTime, default=datetime.utcnow)


class TestUser(Base):
    __tablename__ = "testusers"

    id = Column(Integer, primary_key=True, index=True)
    email = Column(String(255), unique=True, nullable=False, index=True)  # indexed + unique
    name = Column(String(100), nullable=True)  # optional name
    description = Column(Text, nullable=True)  # long text allowed
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
