#!/usr/bin/env python3
"""
Simple script to create database tables
"""

from dotenv import load_dotenv
from db import engine, Base
from models import User, TestUser

# Load environment variables
load_dotenv()

def create_tables():
    """Create all database tables"""
    try:
        Base.metadata.create_all(bind=engine)
        print("✅ Database tables created successfully!")
        print("📋 Tables created:")
        print("   - users")
        print("   - testusers")
    except Exception as e:
        print(f"❌ Error creating tables: {e}")

if __name__ == "__main__":
    create_tables()
