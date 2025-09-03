from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, declarative_base
import os
from urllib.parse import quote_plus
from dotenv import load_dotenv

load_dotenv()

DB_SERVER = os.getenv("AZURE_SQL_SERVER")
DB_PORT = 1433
DB_NAME = os.getenv("AZURE_SQL_DATABASE")
DB_USER = os.getenv("AZURE_SQL_USERNAME")
DB_PASS = os.getenv("AZURE_SQL_PASSWORD")  # keep password in .env

# Encode special characters in password
DB_PASS_ESCAPED = quote_plus(DB_PASS)

sqlalchemy_url = f"mssql+pymssql://{DB_USER}:{DB_PASS_ESCAPED}@{DB_SERVER}:{DB_PORT}/{DB_NAME}"

engine = create_engine(sqlalchemy_url, pool_pre_ping=True)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

def get_db():
    db = SessionLocal()
    try:
        yield db
        db.commit()
    except:
        db.rollback()
        raise
    finally:
        db.close()