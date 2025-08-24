from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import RedirectResponse
import httpx
import urllib.parse
from dotenv import load_dotenv
import os
load_dotenv()

app = FastAPI()

GOOGLE_CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID")
GOOGLE_CLIENT_SECRET = os.getenv("GOOGLE_CLIENT_SECRET")
GOOGLE_REDIRECT_URI = os.getenv("GOOGLE_REDIRECT_URI")

GOOGLE_AUTH_ENDPOINT = "https://accounts.google.com/o/oauth2/v2/auth"
GOOGLE_TOKEN_ENDPOINT = "https://oauth2.googleapis.com/token"
GOOGLE_USERINFO_ENDPOINT = "https://www.googleapis.com/oauth2/v2/userinfo"

SCOPES = ["openid", "email", "profile"]

@app.get("/auth/google/login")
def login():
    # Prepare the Google OAuth URL
    auth_url = (
        GOOGLE_AUTH_ENDPOINT
        + "?"
        + urllib.parse.urlencode(
            {
                "client_id": GOOGLE_CLIENT_ID,
                "redirect_uri": GOOGLE_REDIRECT_URI,
                "response_type": "code",
                "scope": " ".join(SCOPES),
                "access_type": "offline",
                "prompt": "consent",
            }
        )
    )
    return RedirectResponse(auth_url)

@app.get("/auth/google/callback")
async def callback(code: str):
    # Exchange code for tokens
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

    # Use access token to get user info
    async with httpx.AsyncClient() as client:
        userinfo_response = await client.get(
            GOOGLE_USERINFO_ENDPOINT,
            headers={"Authorization": f"Bearer {access_token}"},
        )
    if userinfo_response.status_code != 200:
        raise HTTPException(status_code=400, detail="Failed to fetch user info")

    user_info = userinfo_response.json()
    return user_info
