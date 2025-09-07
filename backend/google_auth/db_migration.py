from database import Base, engine, SessionLocal
from models import User

try:
    Base.metadata.create_all(bind=engine)
    print("Tables created!")

    # Create a session
    db = SessionLocal()

    # Run a test query
    spaces = db.query(User).all()
    print("Spaces:", spaces)

    db.close()
except Exception as e:
    print(e)
