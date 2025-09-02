from database import SessionLocal, Base, engine
from models import User, TestUser

Base.metadata.create_all(bind=engine)