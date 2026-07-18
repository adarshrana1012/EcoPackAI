"""
settings.py — Environment Configuration Management (Prompt 40)
================================================================

Pydantic-settings based configuration with environment variable loading,
defaults, and connection validation.

Usage
-----
    from src.settings import get_settings
    settings = get_settings()
    settings.validate_connections()

Author: EcoPackAI Team
"""

from __future__ import annotations

import json
import logging
from functools import lru_cache
from typing import Any, Dict, Optional

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# Custom Exception
# ═══════════════════════════════════════════════════════════════════════════

class ConfigurationError(Exception):
    """Raised when application configuration is invalid or a required
    external service is unreachable at startup."""
    pass


# ═══════════════════════════════════════════════════════════════════════════
# Settings Class
# ═══════════════════════════════════════════════════════════════════════════

class Settings(BaseSettings):
    """Application settings loaded from environment variables.

    All settings have sensible defaults for local development.
    In production, they should be set via environment variables,
    Kubernetes ConfigMaps/Secrets, or ``.env`` files.

    Attributes
    ----------
    DATABASE_URL : str
        PostgreSQL connection string.
    REDIS_URL : str
        Redis connection string.
    MODEL_PATH : str
        Path to model artifacts directory.
    LOG_LEVEL : str
        Logging level (debug, info, warning, error, critical).
    FEATURE_FLAGS : dict
        JSON dictionary of feature toggles.
    API_RATE_LIMIT : int
        Max requests per minute per client.
    APP_ENV : str
        Environment name (development, staging, production).
    """

    # Database
    DATABASE_URL: str = Field(
        default="postgresql://ecopack:ecopack_secret@localhost:5432/ecopackai",
        description="PostgreSQL connection string",
    )

    # Redis
    REDIS_URL: str = Field(
        default="redis://localhost:6379/0",
        description="Redis connection string for caching and Celery broker",
    )

    # Model artifacts
    MODEL_PATH: str = Field(
        default="./models",
        description="Path to model artifacts directory",
    )

    # Logging
    LOG_LEVEL: str = Field(
        default="info",
        description="Logging level",
    )

    # Feature flags (JSON dict)
    FEATURE_FLAGS: Dict[str, Any] = Field(
        default_factory=lambda: {
            "enable_rl_packing": False,
            "enable_ab_testing": True,
            "enable_prometheus": True,
        },
        description="Feature toggle flags as JSON dict",
    )

    # Rate limiting
    API_RATE_LIMIT: int = Field(
        default=100,
        ge=1,
        le=10000,
        description="Max requests per minute per client",
    )

    # Environment
    ENV: str = Field(
        default="development",
        description="Application environment (development, staging, production)",
    )

    FRONTEND_URL: str = Field(
        default="http://localhost",
        description="Frontend application URL",
    )

    SENTRY_DSN: Optional[str] = Field(
        default=None,
        description="Error tracking DSN",
    )

    OTLP_ENDPOINT: str = Field(
        default="http://jaeger:4317",
        description="OpenTelemetry exporter endpoint",
    )

    # JWT Authentication (Phase 8)
    JWT_SECRET_KEY: str = Field(
        default="ecopackai-dev-secret-CHANGE-IN-PRODUCTION-use-strong-random-key",
        description="HMAC secret key for JWT signing (use RS256 + keypair in production)",
    )

    JWT_ALGORITHM: str = Field(
        default="HS256",
        description="JWT signing algorithm (HS256 for dev, RS256 for prod)",
    )

    JWT_EXPIRE_MINUTES: int = Field(
        default=60,
        ge=1,
        le=10080,  # 1 week max
        description="JWT access token lifetime in minutes",
    )

    # CORS (Phase 8)
    CORS_ORIGINS: str = Field(
        default="*",
        description="Comma-separated list of allowed CORS origins, or '*' for all",
    )

    # --- Validators ---

    @field_validator("LOG_LEVEL")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        allowed = {"debug", "info", "warning", "error", "critical"}
        if v.lower() not in allowed:
            raise ValueError(f"LOG_LEVEL must be one of {allowed}, got '{v}'")
        return v.lower()

    @field_validator("ENV")
    @classmethod
    def validate_env(cls, v: str) -> str:
        allowed = {"development", "staging", "production", "testing"}
        if v.lower() not in allowed:
            raise ValueError(f"ENV must be one of {allowed}, got '{v}'")
        return v.lower()

    @field_validator("FEATURE_FLAGS", mode="before")
    @classmethod
    def parse_feature_flags(cls, v: Any) -> Dict[str, Any]:
        if isinstance(v, str):
            try:
                return json.loads(v)
            except json.JSONDecodeError:
                raise ValueError(f"FEATURE_FLAGS must be valid JSON, got: {v}")
        return v

    # --- Configuration ---

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "case_sensitive": True,
        "extra": "ignore",
    }

    # --- Methods and Properties ---

    @property
    def is_production(self) -> bool:
        return self.ENV == "production"

    @property
    def cors_origins_list(self) -> list[str]:
        if not self.CORS_ORIGINS or self.CORS_ORIGINS.strip() == "*":
            return ["*"]
        return [o.strip() for o in self.CORS_ORIGINS.split(",") if o.strip()]

    def is_feature_enabled(self, flag: str) -> bool:
        """Check if a feature flag is enabled.

        Parameters
        ----------
        flag : str
            Feature flag name.

        Returns
        -------
        bool
        """
        return bool(self.FEATURE_FLAGS.get(flag, False))

    def validate_connections(self) -> Dict[str, bool]:
        """Validate connectivity to external services.

        Checks PostgreSQL and Redis connections. Raises
        ``ConfigurationError`` with clear messages if a required
        service is unreachable.

        Returns
        -------
        dict[str, bool]
            Connection status for each service.

        Raises
        ------
        ConfigurationError
            If any required connection fails.
        """
        results: Dict[str, bool] = {}
        errors: list[str] = []

        # --- PostgreSQL ---
        results["postgres"] = self._check_postgres()
        if not results["postgres"]:
            errors.append(
                f"PostgreSQL connection failed. "
                f"URL: {self._mask_url(self.DATABASE_URL)}. "
                f"Ensure the database is running and credentials are correct."
            )

        # --- Redis ---
        results["redis"] = self._check_redis()
        if not results["redis"]:
            errors.append(
                f"Redis connection failed. "
                f"URL: {self._mask_url(self.REDIS_URL)}. "
                f"Ensure Redis is running on the configured host/port."
            )

        # --- Production specific checks ---
        if self.is_production:
            if self.JWT_SECRET_KEY == "dev-secret-change-in-production" or "dev" in self.JWT_SECRET_KEY:
                errors.append("Insecure JWT_SECRET_KEY used in production environment.")
            if self.SENTRY_DSN is None:
                logger.warning("SENTRY_DSN is not configured in production environment.")

        if errors:
            msg = (
                "Configuration validation failed:\n"
                + "\n".join(f"  • {e}" for e in errors)
            )
            logger.error(msg)
            if self.is_production:
                raise ConfigurationError(msg)
            else:
                logger.warning(
                    "Non-production environment — continuing despite "
                    "connection failures or config errors."
                )

        logger.info("Connection validation: %s", results)
        return results

    def _check_postgres(self) -> bool:
        """Attempt a PostgreSQL connection."""
        try:
            from sqlalchemy import create_engine, text
            engine = create_engine(self.DATABASE_URL, pool_pre_ping=True)
            with engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            engine.dispose()
            logger.info("PostgreSQL connection OK.")
            return True
        except Exception as e:
            logger.warning("PostgreSQL check failed: %s", e)
            return False

    def _check_redis(self) -> bool:
        """Attempt a Redis ping."""
        try:
            import redis
            r = redis.from_url(self.REDIS_URL, socket_timeout=3)
            r.ping()
            r.close()
            logger.info("Redis connection OK.")
            return True
        except Exception as e:
            logger.warning("Redis check failed: %s", e)
            return False

    @staticmethod
    def _mask_url(url: str) -> str:
        """Mask password in connection URL for safe logging."""
        try:
            from urllib.parse import urlparse, urlunparse
            parsed = urlparse(url)
            if parsed.password:
                masked = parsed._replace(
                    netloc=f"{parsed.username}:***@{parsed.hostname}"
                    + (f":{parsed.port}" if parsed.port else "")
                )
                return urlunparse(masked)
        except Exception:
            pass
        return "***masked***"


# ═══════════════════════════════════════════════════════════════════════════
# Singleton accessor
# ═══════════════════════════════════════════════════════════════════════════

@lru_cache()
def get_settings() -> Settings:
    """Get the cached application settings singleton."""
    settings = Settings()
    logger.info(
        "Settings loaded: env=%s, log=%s, rate_limit=%d, "
        "features=%s",
        settings.ENV, settings.LOG_LEVEL,
        settings.API_RATE_LIMIT,
        list(k for k, v in settings.FEATURE_FLAGS.items() if v),
    )
    return settings
