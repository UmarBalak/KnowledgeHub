from fastapi import FastAPI, Depends, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
import httpx, os, urllib.parse
from dotenv import load_dotenv
from db import get_db, engine, Base
from models import User
from sqlalchemy.orm import Session
import jwt
from datetime import datetime, timedelta
import logging
from starlette.responses import JSONResponse, Response
from starlette import status

# Set up logging
logging.basicConfig(level=logging.INFO)

load_dotenv()
Base.metadata.create_all(bind=engine)

app = FastAPI()

GOOGLE_CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID")
GOOGLE_CLIENT_SECRET = os.getenv("GOOGLE_CLIENT_SECRET")
GOOGLE_REDIRECT_URI = os.getenv("GOOGLE_REDIRECT_URI")
FRONTEND_URL = os.getenv("FRONTEND_URL")
SECRET_KEY = os.getenv("SECRET_KEY")
ALGORITHM = "HS256"

GOOGLE_AUTH_ENDPOINT = "https://accounts.google.com/o/oauth2/v2/auth"
GOOGLE_TOKEN_ENDPOINT = "https://oauth2.googleapis.com/token"
GOOGLE_USERINFO_ENDPOINT = "https://www.googleapis.com/oauth2/v2/userinfo"
SCOPES = ["openid", "email", "profile"]

origins = [
    "http://localhost:3000",
    FRONTEND_URL,
]

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["GET"],
    allow_headers=["Authorization", "Content-Type"],
)

# JWT utility
def create_access_token(data: dict, expires_delta: timedelta = timedelta(hours=1)):
    to_encode = data.copy()
    expire = datetime.utcnow() + expires_delta
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

def get_current_user(request: Request, db: Session = Depends(get_db)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        token = request.headers.get("Authorization")
        if token and token.startswith("Bearer "):
            token = token.split(" ")[1]
        else:
            raise credentials_exception
        
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        user_id: str = payload.get("sub")
        if user_id is None:
            raise credentials_exception
        
    except jwt.PyJWTError:
        raise credentials_exception
    
    user = db.query(User).filter(User.id == user_id).first()
    if user is None:
        raise credentials_exception
    return user

@app.get("/")
async def root():
    return {"message": "Google OAuth2 Authentication Service is running."}

@app.get("/health", response_class=JSONResponse)
async def health_check():
    """
    Health check endpoint to verify service availability.
    """
    return {"status": "healthy"}

@app.head("/health")
async def health_check_monitor():
    """
    Health check endpoint to verify service availability.
    """
    return Response(status_code=200)

@app.get("/auth/google/login")
def login():
    logging.info("Initiating Google OAuth2 login")
    auth_url = (
        GOOGLE_AUTH_ENDPOINT
        + "?"
        + urllib.parse.urlencode({
            "client_id": GOOGLE_CLIENT_ID,
            "redirect_uri": GOOGLE_REDIRECT_URI,
            "response_type": "code",
            "scope": " ".join(SCOPES),
            "access_type": "offline",
            "prompt": "consent",
        })
    )
    logging.info(f"Redirecting to Google OAuth2 URL: {auth_url}")
    return RedirectResponse(auth_url)

@app.get("/auth/google/callback")
async def callback(code: str, db: Session = Depends(get_db)):
    logging.info(f"Entered callback with code: {code}")
    async with httpx.AsyncClient() as client:
        token_response = await client.post(
            GOOGLE_TOKEN_ENDPOINT,
            data={
                "code": code,
                "client_id": GOOGLE_CLIENT_ID,
                "client_secret": GOOGLE_CLIENT_SECRET,
                "redirect_uri": GOOGLE_REDIRECT_URI,
                "grant_type": "authorization_code",
            },
            headers={"Content-Type": "application/x-www-form-urlencoded"},
        )
    if token_response.status_code != 200:
        raise HTTPException(status_code=400, detail="Failed to get token")
    tokens = token_response.json()
    access_token = tokens.get("access_token")
    
    async with httpx.AsyncClient() as client:
        userinfo_response = await client.get(
            GOOGLE_USERINFO_ENDPOINT, headers={"Authorization": f"Bearer {access_token}"}
        )
    if userinfo_response.status_code != 200:
        raise HTTPException(status_code=400, detail="Failed to fetch user info")
    
    user_info = userinfo_response.json()
    logging.info("Fetched user info successfully")

    user = db.query(User).filter(User.id == user_info["id"]).first()
    logging.info(f"User lookup result: {user}")

    if not user:
        logging.info("User not found. Creating new user in the database")
        user = User(
            id=user_info["id"],
            email=user_info["email"],
            name=user_info.get("name", ""),
            picture=user_info.get("picture", "")
        )
        db.add(user)
        db.commit()
        db.refresh(user)
        logging.info(f"Created new user: {user.email}")

    # Create JWT for user and redirect to frontend with token
    token = create_access_token({"sub": user.id, "email": user.email, "name": user.name})
    # redirect_uri = f"{FRONTEND_URL}/callback?token={token}"
    redirect_uri = f"http://localhost:3000/callback?token={token}"
    logging.info(f"Redirecting to: {redirect_uri}")
    return RedirectResponse(redirect_uri)

@app.get("/auth/google/home")
async def get_home_data(current_user: User = Depends(get_current_user)):
    # This endpoint is now protected.
    # It will only return if the user has a valid JWT token.
    return {"message": f"Welcome, {current_user.name}!"}