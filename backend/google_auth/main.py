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
from fastapi import Response, Cookie
from fastapi.responses import RedirectResponse

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

is_production = True

origins = [
    "http://localhost:3000",
    "https://localhost:3000",
    FRONTEND_URL,
]

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["Authorization", "Content-Type"],
    expose_headers=["Set-Cookie"],
)

# JWT utility
def create_access_token(data: dict, expires_delta: timedelta = timedelta(hours=1)):
    to_encode = data.copy()
    expire = datetime.utcnow() + expires_delta
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

def get_current_user_from_cookie(auth_token: str = Cookie(None), db: Session = Depends(get_db)):
    if not auth_token:
        raise HTTPException(status_code=401, detail="Not authenticated")
    
    try:
        payload = jwt.decode(auth_token, SECRET_KEY, algorithms=[ALGORITHM])
        user_id: str = payload.get("sub")
        if user_id is None:
            raise HTTPException(status_code=401, detail="Invalid token")
        
        user = db.query(User).filter(User.google_id == user_id).first()
        if user is None:
            raise HTTPException(status_code=401, detail="User not found")
        
        return user
    except jwt.PyJWTError:
        raise HTTPException(status_code=401, detail="Invalid token")

@app.get("/")
async def root():
    return {"message": "Google OAuth2 Authentication Service is running."}

@app.get("/health", response_class=JSONResponse)
async def health_check():
    return {"status": "healthy"}

@app.head("/health")
async def health_check_monitor():
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
    
    # Get Google tokens
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
        logging.error(f"Token response error: {token_response.status_code} - {token_response.text}")
        raise HTTPException(status_code=400, detail="Failed to get token")
    
    tokens = token_response.json()
    access_token = tokens.get("access_token")
    
    # Get user info from Google
    async with httpx.AsyncClient() as client:
        userinfo_response = await client.get(
            GOOGLE_USERINFO_ENDPOINT, 
            headers={"Authorization": f"Bearer {access_token}"}
        )
    
    if userinfo_response.status_code != 200:
        logging.error(f"Userinfo response error: {userinfo_response.status_code}")
        raise HTTPException(status_code=400, detail="Failed to fetch user info")
    
    user_info = userinfo_response.json()
    logging.info("Fetched user info successfully")

    # Find or create user
    user = db.query(User).filter(User.google_id == user_info["id"]).first()
    logging.info(f"User lookup result: {user}")

    if not user:
        logging.info("User not found. Creating new user in the database")
        user = User(
            google_id=user_info["id"],
            email=user_info["email"],
            name=user_info.get("name", ""),
            picture=user_info.get("picture", "")
        )
        db.add(user)
        db.commit()
        db.refresh(user)
        logging.info(f"Created new user: {user.email}")

    # Create JWT token with longer expiry
    token = create_access_token(
        {"sub": user.google_id, "email": user.email, "name": user.name},
        expires_delta=timedelta(hours=24)
    )

    # Redirect to frontend
    if is_production:
        redirect_uri = f"{FRONTEND_URL}/home"
    else:
        redirect_uri = "http://localhost:3000/home"
        
    logging.info(f"Setting cookie and redirecting to: {redirect_uri}")
    
    # Create RedirectResponse and set cookie on it
    response = RedirectResponse(url=redirect_uri, status_code=302)
    
    # Set HTTP-only cookie with proper settings
    response.set_cookie(
        key="auth_token",
        value=token,
        httponly=True,
        secure=True,  # Set to True for production
        samesite="lax",  # Can use lax for same-protocol
        max_age=86400,
        path="/",
    )
    
    logging.info(f"Cookie set for user: {user.email}")
    return response

@app.get("/auth/google/home")
async def get_home_data(current_user: User = Depends(get_current_user_from_cookie)):
    # Protected endpoint using cookie authentication
    return {"message": f"Welcome, {current_user.name}!"}

@app.get("/auth/status")
async def check_auth_status(request: Request, auth_token: str = Cookie(None)):
    logging.info(f"Auth status check - Cookie present: {auth_token is not None}")
    logging.info(f"All cookies: {request.cookies}")
    
    if not auth_token:
        logging.warning("No auth token found in cookies")
        raise HTTPException(status_code=401, detail="Not authenticated")
    
    try:
        # Validate JWT token
        payload = jwt.decode(auth_token, SECRET_KEY, algorithms=[ALGORITHM])
        logging.info(f"Token validated for user: {payload.get('email')}")
        return {"authenticated": True, "user": payload}
    except jwt.ExpiredSignatureError:
        logging.warning("Token expired")
        raise HTTPException(status_code=401, detail="Token expired")
    except jwt.PyJWTError as e:
        logging.warning(f"Invalid token: {e}")
        raise HTTPException(status_code=401, detail="Invalid token")

@app.post("/auth/logout")
async def logout(response: Response):
    response.delete_cookie("auth_token", path="/")
    return {"message": "Logged out successfully"}