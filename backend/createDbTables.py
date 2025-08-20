import os
from dotenv import load_dotenv

# Load environment variables (including DATABASE_URL)
load_dotenv()

from database import Base, engine
# Import all your models so SQLAlchemy knows about them
from models import User, Document, DocumentChunk, ReportedContent

def create_tables():
    print("Attempting to create database tables on Neon DB...")
    Base.metadata.create_all(bind=engine)
    print("Database tables created or already exist on Neon DB.")

if __name__ == "__main__":
    # Ensure your .env file is set up with DATABASE_URL
    if not os.getenv("DATABASE_URL"):
        print("Error: DATABASE_URL environment variable is not set.")
        print("Please set it in your .env file with your Neon DB connection string.")
        exit(1)
    create_tables()