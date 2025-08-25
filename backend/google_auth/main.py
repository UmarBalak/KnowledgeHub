from fastapi import FastAPI, Depends, Request, HTTPException
from fastapi.responses import RedirectResponse
import httpx, os, urllib.parse
from dotenv import load_dotenv
from db import get_db, engine, Base
from models import User
from sqlalchemy.orm import Session
import jwt
from datetime import datetime, timedelta

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

# JWT utility
def create_access_token(data: dict, expires_delta: timedelta = timedelta(hours=1)):
    to_encode = data.copy()
    expire = datetime.utcnow() + expires_delta
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

@app.get("/auth/google/login")
def login():
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
    return RedirectResponse(auth_url)

@app.get("/auth/google/callback")
async def callback(code: str, db: Session = Depends(get_db)):
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
    
    user = db.query(User).filter(User.id == user_info["id"]).first()
    if not user:
        user = User(
            id=user_info["id"],
            email=user_info["email"],
            name=user_info.get("name", ""),
            picture=user_info.get("picture", "")
        )
        db.add(user)
        db.commit()
        db.refresh(user)

    # Create JWT for user and redirect to frontend with token
    token = create_access_token({"sub": user.id, "email": user.email, "name": user.name})
    redirect_uri = f"{FRONTEND_URL}?token={token}"
    return RedirectResponse(redirect_uri)
