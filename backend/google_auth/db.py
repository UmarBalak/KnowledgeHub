# db.py
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import os
from dotenv import load_dotenv # Don't forget to import load_dotenv

load_dotenv() # Load environment variables from .env file

DATABASE_URL = os.getenv("DATABASE_URL")

if not DATABASE_URL:
    raise ValueError("DATABASE_URL environment variable is not set. Please set it in your .env file.")

# Neon DB often requires SSL, which is handled by the `sslmode=require` in the connection string.
# No extra `connect_args` are usually needed here if `sslmode=require` is in your URL.
engine = create_engine(
    DATABASE_URL,
    pool_pre_ping=True,  # Validates connections before use
    pool_recycle=300,    # Recreate connections every 5 minutes
    pool_timeout=20,     # Wait 20 seconds for connection
    max_overflow=0,      # Don't create extra connections beyond pool_size
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

def get_db():
    """
    Dependency for FastAPI to get a database session.
    Ensures the session is closed after the request is processed.
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()