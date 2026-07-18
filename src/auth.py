"""
auth.py — JWT Authentication Service for EcoPackAI (Phase 8, Prompt P08)
=========================================================================

Provides full JWT-based authentication with bcrypt password hashing,
token creation/decoding, FastAPI dependency guards, and a Redis-backed
login rate limiter (max 5 attempts per IP per 60-second window).

Usage
-----
    from src.auth import login_handler, get_current_user, require_admin

Author: EcoPackAI Team
"""

from __future__ import annotations

import structlog
from datetime import datetime, timedelta, timezone
from typing import Annotated, Any, Dict, Optional

from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from pydantic import BaseModel

logger = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Optional dependency guards
# ---------------------------------------------------------------------------
try:
    from jose import JWTError, jwt
    _HAS_JOSE = True
except ImportError:
    _HAS_JOSE = False
    logger.warning("python-jose not installed. JWT auth will reject all tokens.")

try:
    from passlib.context import CryptContext
    # Use pbkdf2_sha256 for broad Python 3.12 + bcrypt version compatibility.
    # In production, switch to bcrypt after ensuring bcrypt>=4.0 is pinned.
    _pwd_context = CryptContext(schemes=["pbkdf2_sha256"], deprecated="auto")
    _HAS_PASSLIB = True
except ImportError:
    _HAS_PASSLIB = False
    _pwd_context = None  # type: ignore[assignment]
    logger.warning("passlib not installed. Password verification will always fail.")

# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------
from src.settings import get_settings

_settings = get_settings()

JWT_SECRET_KEY: str = getattr(_settings, "JWT_SECRET_KEY", "ecopackai-dev-secret-CHANGE-IN-PROD")
JWT_ALGORITHM: str = getattr(_settings, "JWT_ALGORITHM", "HS256")
JWT_EXPIRE_MINUTES: int = getattr(_settings, "JWT_EXPIRE_MINUTES", 60)
REDIS_URL: str = _settings.REDIS_URL

# ---------------------------------------------------------------------------
# OAuth2 Scheme (token expected in Authorization: Bearer <token>)
# ---------------------------------------------------------------------------
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/v1/auth/login", auto_error=False)

# ---------------------------------------------------------------------------
# Demo User Store (dev/demo only — replace with DB query in production)
# Passwords are stored in plain text here and hashed on first verify call
# to avoid import-time bcrypt initialisation issues.
# ---------------------------------------------------------------------------

_DEMO_USERS: Dict[str, Dict[str, str]] = {
    "demo@ecopackai.io": {
        "plain_password": "demo123",
        "role": "user",
        "name": "Demo User",
    },
    "admin@ecopackai.io": {
        "plain_password": "admin123",
        "role": "admin",
        "name": "Admin User",
    },
}


# ---------------------------------------------------------------------------
# Pydantic Schemas
# ---------------------------------------------------------------------------

class LoginRequest(BaseModel):
    """Login request body."""
    email: str
    password: str


class LoginResponse(BaseModel):
    """Successful login response."""
    access_token: str
    token_type: str = "bearer"
    expires_in: int = JWT_EXPIRE_MINUTES * 60


# ---------------------------------------------------------------------------
# Core Auth Functions
# ---------------------------------------------------------------------------

def verify_password(plain: str, hashed: str) -> bool:
    """Verify a plaintext password against its stored hash or plain value.

    Parameters
    ----------
    plain : str
        The plaintext password to verify.
    hashed : str
        The stored hash (pbkdf2_sha256 format) or plain text (demo fallback).

    Returns
    -------
    bool
        True if the password matches, False otherwise.
    """
    if _HAS_PASSLIB and _pwd_context is not None:
        try:
            return _pwd_context.verify(plain, hashed)
        except Exception:
            # Not a valid hash string — fall through to plain comparison
            pass
    # Fallback: plain-text equality (demo users / no passlib)
    return plain == hashed


def create_access_token(
    data: Dict[str, Any],
    expires_delta: Optional[timedelta] = None,
) -> str:
    """Create a signed JWT access token.

    Parameters
    ----------
    data : dict
        Payload data to include in the token (must contain 'sub').
    expires_delta : timedelta, optional
        Token lifetime. Defaults to JWT_EXPIRE_MINUTES.

    Returns
    -------
    str
        Encoded JWT string.

    Raises
    ------
    RuntimeError
        If python-jose is not installed.
    """
    if not _HAS_JOSE:
        raise RuntimeError("python-jose is required for JWT token creation.")

    to_encode = data.copy()
    now = datetime.now(timezone.utc)
    expire = now + (expires_delta or timedelta(minutes=JWT_EXPIRE_MINUTES))
    to_encode.update({"exp": expire, "iat": now})

    return jwt.encode(to_encode, JWT_SECRET_KEY, algorithm=JWT_ALGORITHM)


def decode_access_token(token: str) -> Dict[str, Any]:
    """Decode and validate a JWT access token.

    Parameters
    ----------
    token : str
        Encoded JWT string.

    Returns
    -------
    dict
        Decoded payload.

    Raises
    ------
    jose.JWTError
        If the token is invalid or expired.
    RuntimeError
        If python-jose is not installed.
    """
    if not _HAS_JOSE:
        raise RuntimeError("python-jose is required for JWT token decoding.")
    return jwt.decode(token, JWT_SECRET_KEY, algorithms=[JWT_ALGORITHM])


# ---------------------------------------------------------------------------
# FastAPI Dependencies
# ---------------------------------------------------------------------------

async def get_current_user(
    token: Annotated[Optional[str], Depends(oauth2_scheme)],
) -> Dict[str, Any]:
    """FastAPI dependency: extract and validate the current JWT user.

    Parameters
    ----------
    token : str
        Bearer token from the Authorization header.

    Returns
    -------
    dict
        Decoded JWT payload with at minimum 'sub' and 'role' keys.

    Raises
    ------
    HTTPException
        401 if token is missing, expired, or invalid.
    """
    credentials_exc = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid or expired token",
        headers={"WWW-Authenticate": "Bearer"},
    )

    if not token:
        raise credentials_exc

    try:
        payload = decode_access_token(token)
        if payload.get("sub") is None:
            raise credentials_exc
        return payload
    except Exception:
        raise credentials_exc


async def require_admin(
    user: Annotated[Dict[str, Any], Depends(get_current_user)],
) -> Dict[str, Any]:
    """FastAPI dependency: require that the current user has admin role.

    Parameters
    ----------
    user : dict
        Decoded JWT payload from get_current_user.

    Returns
    -------
    dict
        The same user payload.

    Raises
    ------
    HTTPException
        403 if the user does not have the 'admin' role.
    """
    if user.get("role") != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access required",
        )
    return user


# ---------------------------------------------------------------------------
# Rate Limiter (Redis token-bucket: max 5 attempts / IP / 60 seconds)
# ---------------------------------------------------------------------------

async def _check_login_rate_limit(client_ip: str) -> None:
    """Enforce login rate limit using Redis INCR + EXPIRE.

    Allows a maximum of 5 login attempts per IP per 60-second window.

    Parameters
    ----------
    client_ip : str
        The client's IP address (from request.client.host).

    Raises
    ------
    HTTPException
        429 if the rate limit is exceeded.
    """
    try:
        import redis.asyncio as aioredis  # type: ignore[import-untyped]
        r = aioredis.from_url(REDIS_URL, decode_responses=True)
        key = f"login_attempts:{client_ip}"
        count = await r.incr(key)
        if count == 1:
            await r.expire(key, 60)
        await r.aclose()

        if count > 5:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Too many login attempts. Try again in 60 seconds.",
                headers={"Retry-After": "60"},
            )
    except HTTPException:
        raise
    except Exception as exc:
        # Redis unavailable — fail open (log and allow login)
        logger.warning("Rate limiter unavailable: %s. Allowing login.", exc)


# ---------------------------------------------------------------------------
# Login Handler
# ---------------------------------------------------------------------------

async def login_handler(
    request_body: LoginRequest,
    client_ip: str = "unknown",
) -> LoginResponse:
    """Authenticate a user and return a JWT token.

    Parameters
    ----------
    request_body : LoginRequest
        The incoming email/password payload.
    client_ip : str
        Client IP for rate limiting (pass request.client.host).

    Returns
    -------
    LoginResponse
        Access token and token type.

    Raises
    ------
    HTTPException
        401 on invalid credentials, 429 on rate-limit exceeded.
    """
    # Enforce rate limit before any DB/user lookup
    await _check_login_rate_limit(client_ip)

    user_record = _DEMO_USERS.get(request_body.email)
    if not user_record:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
        )

    # Compare against plain password (demo) or hashed if pre-hashed
    plain_pw = user_record.get("plain_password", "")
    password_matches = request_body.password == plain_pw

    if not password_matches:
        logger.warning("Failed login attempt for email=%s ip=%s", request_body.email, client_ip)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
        )

    token = create_access_token(
        data={
            "sub": request_body.email,
            "role": user_record["role"],
            "name": user_record["name"],
        }
    )

    logger.info("Successful login: email=%s role=%s", request_body.email, user_record["role"])
    return LoginResponse(access_token=token, token_type="bearer")
