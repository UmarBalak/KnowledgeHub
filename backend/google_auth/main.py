from fastapi import FastAPI, Depends, Request, HTTPException, Cookie, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
import httpx, os, urllib.parse, jwt, logging
from dotenv import load_dotenv
from sqlalchemy.orm import Session
from datetime import datetime, timedelta
from starlette.responses import JSONResponse
from starlette import status

from db import get_db, engine, Base
from models import User

# ---------- Setup ----------
logging.basicConfig(level=logging.INFO)
load_dotenv()
Base.metadata.create_all(bind=engine)

app = FastAPI()

GOOGLE_CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID")
GOOGLE_CLIENT_SECRET = os.getenv("GOOGLE_CLIENT_SECRET")
GOOGLE_REDIRECT_URI = os.getenv("GOOGLE_REDIRECT_URI")
FRONTEND_URL = os.getenv("FRONTEND_URL")  # e.g. https://your-frontend.com
SECRET_KEY = os.getenv("SECRET_KEY")
ALGORITHM = "HS256"

GOOGLE_AUTH_ENDPOINT = "https://accounts.google.com/o/oauth2/v2/auth"
GOOGLE_TOKEN_ENDPOINT = "https://oauth2.googleapis.com/token"
GOOGLE_USERINFO_ENDPOINT = "https://www.googleapis.com/oauth2/v2/userinfo"
SCOPES = ["openid", "email", "profile"]

# In production both FE/BE are HTTPS → cookies must be Secure + SameSite=None for cross-site
is_production = True

# ---------- CORS ----------
# Must be the exact deployed frontend origin for cookies to flow
app.add_middleware(
    CORSMiddleware,
    allow_origins=[FRONTEND_URL],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["Authorization", "Content-Type"],
    expose_headers=["Set-Cookie"],
)

# ---------- Utils ----------
def create_access_token(data: dict, expires_delta: timedelta = timedelta(hours=24)):
    to_encode = data.copy()
    expire = datetime.utcnow() + expires_delta
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

def get_current_user_from_cookie(auth_token: str = Cookie(None), db: Session = Depends(get_db)):
    if not auth_token:
        raise HTTPException(status_code=401, detail="Not authenticated")
    try:
        payload = jwt.decode(auth_token, SECRET_KEY, algorithms=[ALGORITHM])
        user_id: str = payload.get("sub")
        if not user_id:
            raise HTTPException(status_code=401, detail="Invalid token")
        user = db.query(User).filter(User.google_id == user_id).first()
        if not user:
            raise HTTPException(status_code=401, detail="User not found")
        return user
    except jwt.PyJWTError:
        raise HTTPException(status_code=401, detail="Invalid token")

# ---------- Health ----------
@app.get("/")
async def root():
    return {"message": "Auth service is running."}

@app.get("/health", response_class=JSONResponse)
async def health_check():
    return {"status": "healthy"}

@app.head("/health")
async def health_check_head():
    return Response(status_code=200)

@app.get("/dbhealth", response_class=JSONResponse)
async def db_health_check(db: Session = Depends(get_db)):
    try:
        db.execute("SELECT 1")
        return {"status": "healthy", "database": "connected"}
    except Exception as e:
        raise HTTPException(status_code=503, detail={"status": "unhealthy", "error": str(e)})

@app.head("/dbhealth")
async def db_health_check_head(db: Session = Depends(get_db)):
    try:
        db.execute("SELECT 1")
        return Response(status_code=200)
    except Exception as e:
        return Response(status_code=503)

# ---------- OAuth ----------
@app.get("/auth/google/login")
def login():
    params = {
        "client_id": GOOGLE_CLIENT_ID,
        "redirect_uri": GOOGLE_REDIRECT_URI,
        "response_type": "code",
        "scope": " ".join(SCOPES),
        "access_type": "offline",
        "prompt": "consent",
    }
    return RedirectResponse(GOOGLE_AUTH_ENDPOINT + "?" + urllib.parse.urlencode(params))

@app.get("/auth/google/callback")
async def callback(code: str, db: Session = Depends(get_db)):
    async with httpx.AsyncClient() as client:
        token_resp = await client.post(
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
    if token_resp.status_code != 200:
        raise HTTPException(status_code=400, detail="Failed to get token")
    access_token = token_resp.json().get("access_token")

    async with httpx.AsyncClient() as client:
        userinfo_resp = await client.get(
            GOOGLE_USERINFO_ENDPOINT,
            headers={"Authorization": f"Bearer {access_token}"},
        )
    if userinfo_resp.status_code != 200:
        raise HTTPException(status_code=400, detail="Failed to fetch user info")

    info = userinfo_resp.json()
    user = db.query(User).filter(User.google_id == info["id"]).first()
    if not user:
        user = User(
            google_id=info["id"],
            email=info["email"],
            name=info.get("name", ""),
            picture=info.get("picture", ""),
        )
        db.add(user)
        db.commit()
        db.refresh(user)

    jwt_token = create_access_token({"sub": user.google_id, "email": user.email, "name": user.name})

    # Redirect to frontend and set cookie on the redirect response
    redirect_uri = f"{FRONTEND_URL}/home"
    resp = RedirectResponse(url=redirect_uri, status_code=302)
    resp.set_cookie(
        key="auth_token",
        value=jwt_token,
        httponly=True,
        secure=True,          # required for HTTPS
        samesite="none",      # required for cross-site cookies
        max_age=86400,
        path="/",
    )
    return resp

# ---------- Auth status (used by frontend) ----------
@app.get("/auth/status")
async def check_auth_status(request: Request, auth_token: str = Cookie(None)):
    if not auth_token:
        raise HTTPException(status_code=401, detail="Not authenticated")
    try:
        payload = jwt.decode(auth_token, SECRET_KEY, algorithms=[ALGORITHM])
        return {"authenticated": True, "user": {"email": payload.get("email"), "name": payload.get("name")}}
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired")
    except jwt.PyJWTError:
        raise HTTPException(status_code=401, detail="Invalid token")

# ---------- Logout ----------
@app.post("/auth/logout")
async def logout(response: Response):
    response.delete_cookie(
        key="auth_token",
        path="/",
        secure=True,      # Match the secure flag used when setting the cookie
        samesite="none"   # Match the SameSite attribute used when setting the cookie
    )
    logging.info("User logged out, auth_token cookie deleted")
    return {"message": "Logged out successfully"}