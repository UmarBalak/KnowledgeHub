from fastapi import FastAPI, Depends, Request, HTTPException, Cookie, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
import httpx, os, urllib.parse, jwt, logging
from dotenv import load_dotenv
from sqlalchemy.orm import Session
from sqlalchemy import text
from datetime import datetime, timedelta
from starlette.responses import JSONResponse
from starlette import status

from database import get_db, engine, Base
from models import User, TestUser, RoleEnum

# ---------- Setup ----------
logging.basicConfig(level=logging.INFO)
load_dotenv()

Base.metadata.create_all(bind=engine)

app = FastAPI()


GOOGLE_EXTERNAL_CLIENT_ID = os.getenv("GOOGLE_EXTERNAL_CLIENT_ID")
GOOGLE_EXTERNAL_CLIENT_SECRET = os.getenv("GOOGLE_EXTERNAL_CLIENT_SECRET")
GOOGLE_EXTERNAL_REDIRECT_URI = os.getenv("GOOGLE_EXTERNAL_REDIRECT_URI")

CLG_DOMAIN = os.getenv("CLG_DOMAIN")

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
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["Set-Cookie"],
)


# ---------- Utils ----------
def create_access_token(data: dict, expires_delta: timedelta = timedelta(hours=24)):
    to_encode = data.copy()
    expire = datetime.utcnow() + expires_delta
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

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
        db.execute(text("SELECT 1"))
        return {"status": "healthy", "database": "connected"}
    except Exception as e:
        raise HTTPException(status_code=503, detail={"status": "unhealthy", "error": str(e)})

@app.head("/dbhealth")
async def db_health_check_head(db: Session = Depends(get_db)):
    try:
        db.execute(text("SELECT 1"))
        return Response(status_code=200)
    except Exception as e:
        return Response(status_code=503)

# ---------- WS Token Endpoint ----------
@app.get("/auth/ws-token")
def issue_ws_token(auth_token: str = Cookie(None)):
    """
    Uses HttpOnly auth_token cookie to mint a short-lived WS token.
    """
    if not auth_token:
        raise HTTPException(status_code=401, detail="Not authenticated")

    try:
        payload = jwt.decode(auth_token, SECRET_KEY, algorithms=[ALGORITHM])

        ws_token = create_access_token(
            {
                "sub": payload.get("sub"),
                "name": payload.get("name"),
                "role": payload.get("role"),
                "scope": "ws",
            },
            expires_delta=timedelta(minutes=5),
        )

        return {"token": ws_token}

    except jwt.PyJWTError:
        raise HTTPException(status_code=401, detail="Invalid token")
        
# ---------- OAuth ----------
@app.get("/auth/google/guestlogin")
def guestlogin():
    params = {
        "client_id": GOOGLE_EXTERNAL_CLIENT_ID,
        "redirect_uri": GOOGLE_EXTERNAL_REDIRECT_URI,
        "response_type": "code",
        "scope": " ".join(SCOPES),
        "access_type": "offline",
        "prompt": "consent",
    }
    return RedirectResponse(GOOGLE_AUTH_ENDPOINT + "?" + urllib.parse.urlencode(params))

@app.get("/auth/google/guestcallback")
async def guestcallback(code: str = None, error: str = None, db: Session = Depends(get_db)):
    # Handle OAuth errors from Google
    if error:
        error_mapping = {
            "access_denied": "User denied access to the application",
            "invalid_request": "Invalid request to Google OAuth",
            "unauthorized_client": "Unauthorized client",
            "unsupported_response_type": "Unsupported response type",
            "invalid_scope": "Invalid scope requested",
            "server_error": "Google OAuth server error",
            "temporarily_unavailable": "Google OAuth temporarily unavailable"
        }
        error_message = error_mapping.get(error, f"OAuth error: {error}")
        redirect_uri = f"{FRONTEND_URL}/auth-error?error=oauth_error&error_description={urllib.parse.quote(error_message)}"
        return RedirectResponse(url=redirect_uri, status_code=302)
    
    # Handle missing authorization code
    if not code:
        redirect_uri = f"{FRONTEND_URL}/auth-error?error=missing_code&error_description=No authorization code received from Google"
        return RedirectResponse(url=redirect_uri, status_code=302)

    try:
        # Exchange code for token
        async with httpx.AsyncClient() as client:
            token_resp = await client.post(
                GOOGLE_TOKEN_ENDPOINT,
                data={
                    "code": code,
                    "client_id": GOOGLE_EXTERNAL_CLIENT_ID,
                    "client_secret": GOOGLE_EXTERNAL_CLIENT_SECRET,
                    "redirect_uri": GOOGLE_EXTERNAL_REDIRECT_URI,
                    "grant_type": "authorization_code",
                },
                headers={"Content-Type": "application/x-www-form-urlencoded"},
            )
        
        if token_resp.status_code != 200:
            error_detail = "Failed to exchange authorization code for token"
            try:
                error_data = token_resp.json()
                if "error_description" in error_data:
                    error_detail = error_data["error_description"]
            except:
                pass
            redirect_uri = f"{FRONTEND_URL}/auth-error?error=token_exchange_failed&error_description={urllib.parse.quote(error_detail)}"
            return RedirectResponse(url=redirect_uri, status_code=302)
        
        token_data = token_resp.json()
        access_token = token_data.get("access_token")
        
        if not access_token:
            redirect_uri = f"{FRONTEND_URL}/auth-error?error=no_access_token&error_description=No access token received from Google"
            return RedirectResponse(url=redirect_uri, status_code=302)

        # Fetch user info
        async with httpx.AsyncClient() as client:
            userinfo_resp = await client.get(
                GOOGLE_USERINFO_ENDPOINT,
                headers={"Authorization": f"Bearer {access_token}"},
            )
        
        if userinfo_resp.status_code != 200:
            error_detail = "Failed to fetch user information from Google"
            try:
                error_data = userinfo_resp.json()
                if "error_description" in error_data:
                    error_detail = error_data["error_description"]
            except:
                pass
            redirect_uri = f"{FRONTEND_URL}/auth-error?error=userinfo_failed&error_description={urllib.parse.quote(error_detail)}"
            return RedirectResponse(url=redirect_uri, status_code=302)

        info = userinfo_resp.json()
        
        # Validate required fields
        if not info.get("email"):
            redirect_uri = f"{FRONTEND_URL}/auth-error?error=no_email&error_description=Email address not provided by Google"
            return RedirectResponse(url=redirect_uri, status_code=302)
        
        email = info["email"].lower()

        # ---------- Access Rules ----------
        allowed = False

        # Rule 1: Check if email is in TestUser allowlist (for first-time access)
        test_user = db.query(TestUser).filter(
            TestUser.email == email,
            TestUser.is_active == True
        ).first()
        if test_user:
            allowed = True

        # Rule 2: Check if domain is CLG_DOMAIN
        if CLG_DOMAIN and email.endswith(str(CLG_DOMAIN)):
            allowed = True

        if not allowed:
            redirect_uri = f"{FRONTEND_URL}/signin?error=unauthorized_email"
            return RedirectResponse(url=redirect_uri, status_code=302)

        # ---------- Save or update user in User table ----------
        try:
            # Check if user already exists in User table
            user = db.query(User).filter(User.google_id == info["id"]).first()
            
            if not user:
                # First time user - create in User table
                user = User(
                    google_id=info["id"],
                    email=email,
                    name=info.get("name", ""),
                    picture=info.get("picture", ""),
                    role=RoleEnum.user
                )
                db.add(user)
                db.commit()
                db.refresh(user)
                
                # Optionally remove from TestUser table after successful first login
                # if test_user:
                #     db.delete(test_user)
                #     db.commit()
            else:
                # Existing user - update info
                user.name = info.get("name", user.name)
                user.picture = info.get("picture", user.picture)
                user.role = info.get("role", user.role.value)
                db.commit()
                db.refresh(user)

            jwt_token = create_access_token(
                {"sub": user.google_id, "email": user.email, "name": user.name, "role": user.role}
            )

            # Redirect to frontend and set cookie
            redirect_uri = f"{FRONTEND_URL}/home"
            resp = RedirectResponse(url=redirect_uri, status_code=302)
            resp.set_cookie(
                key="auth_token",
                value=jwt_token,
                httponly=True,
                secure=True,
                samesite="none",
                path="/",
                domain=".onrender.com",
                max_age=86400,
            )

            return resp
            
        except Exception as e:
            logging.error(f"Database error during user creation/update: {str(e)}")
            redirect_uri = f"{FRONTEND_URL}/auth-error?error=database_error&error_description=Failed to save user information"
            return RedirectResponse(url=redirect_uri, status_code=302)
            
    except httpx.RequestError as e:
        logging.error(f"Network error during OAuth process: {str(e)}")
        redirect_uri = f"{FRONTEND_URL}/auth-error?error=network_error&error_description=Network error occurred during authentication"
        return RedirectResponse(url=redirect_uri, status_code=302)
    except Exception as e:
        logging.error(f"Unexpected error during OAuth process: {str(e)}")
        redirect_uri = f"{FRONTEND_URL}/auth-error?error=unexpected_error&error_description=An unexpected error occurred"
        return RedirectResponse(url=redirect_uri, status_code=302)

# ---------- Auth status (used by frontend) ----------
@app.get("/auth/status")
async def check_auth_status(request: Request, auth_token: str = Cookie(None)):
    if not auth_token:
        raise HTTPException(status_code=401, detail="Not authenticated")
    try:
        payload = jwt.decode(auth_token, SECRET_KEY, algorithms=[ALGORITHM])
        return {
            "authenticated": True,
            "user": {
                "id": payload.get("sub"),
                "email": payload.get("email"),
                "name": payload.get("name"),
                "role": payload.get("role"),
            }
        }
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
