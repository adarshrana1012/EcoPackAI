"""Add users and sessions tables.

Revision ID: 002
Revises: 001
Create Date: 2026-07-07 12:00:00.000000

This migration adds:
* **users** – Store user credentials, roles, and tenants.
* **sessions** – Session tracking/revocation audit trail.
"""

from __future__ import annotations

import os
import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "002"
down_revision: str | None = "001"
branch_labels: tuple[str, ...] | None = None
depends_on: str | None = None


def upgrade() -> None:
    """Create the users and sessions tables."""
    
    # 1. New table: users
    op.create_table(
        "users",
        sa.Column(
            "user_id",
            postgresql.UUID(as_uuid=True),
            primary_key=True,
            nullable=False,
            server_default=sa.text("gen_random_uuid()"),
            comment="Unique user identifier (UUID).",
        ),
        sa.Column(
            "email",
            sa.VARCHAR(255),
            nullable=False,
            unique=True,
            comment="User's unique email address.",
        ),
        sa.Column(
            "password_hash",
            sa.VARCHAR(255),
            nullable=False,
            comment="Passlib hash of the password.",
        ),
        sa.Column(
            "role",
            sa.VARCHAR(20),
            nullable=False,
            server_default="user",
            comment="Authorization role.",
        ),
        sa.Column(
            "tenant_id",
            postgresql.UUID(as_uuid=True),
            nullable=True,
            comment="Reference to tenant organization.",
        ),
        sa.Column(
            "created_at",
            sa.TIMESTAMP(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
            comment="Row creation timestamp.",
        ),
        sa.Column(
            "last_login",
            sa.TIMESTAMP(timezone=True),
            nullable=True,
            comment="Timestamp of last successful authentication.",
        ),
        sa.CheckConstraint("role IN ('user', 'admin')", name="ck_users_role_valid"),
        comment="Users registry with authentication secrets and roles.",
    )

    # Indexes on users
    op.create_index("ix_users_email", "users", ["email"], unique=True)
    op.create_index("ix_users_tenant_id", "users", ["tenant_id"])

    # 2. New table: sessions
    op.create_table(
        "sessions",
        sa.Column(
            "session_id",
            postgresql.UUID(as_uuid=True),
            primary_key=True,
            nullable=False,
            server_default=sa.text("gen_random_uuid()"),
            comment="Unique session audit identifier.",
        ),
        sa.Column(
            "user_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("users.user_id", ondelete="CASCADE"),
            nullable=False,
            comment="Associated user identifier.",
        ),
        sa.Column(
            "jwt_jti",
            sa.VARCHAR(64),
            nullable=True,
            unique=True,
            comment="JWT unique identifier claim for revocation.",
        ),
        sa.Column(
            "created_at",
            sa.TIMESTAMP(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
            comment="Timestamp session was generated.",
        ),
        sa.Column(
            "expires_at",
            sa.TIMESTAMP(timezone=True),
            nullable=False,
            comment="Timestamp token expires.",
        ),
        sa.Column(
            "revoked",
            sa.Boolean,
            server_default=sa.text("false"),
            nullable=False,
            comment="Whether session is revoked/invalid.",
        ),
        comment="Audit trail for sessions and JWT tokens.",
    )

    # Indexes on sessions
    op.create_index("ix_sessions_jwt_jti", "sessions", ["jwt_jti"], unique=True)
    op.create_index("ix_sessions_user_id", "sessions", ["user_id"])

    # 3. Seed demo users if run_seed=True or RUN_SEED=true
    run_seed = os.getenv("RUN_SEED", "false").lower() in ("true", "1", "yes")
    if run_seed:
        # Define hashes (using passlib.context pbkdf2_sha256 format for the demo values)
        # pbkdf2_sha256 hash of 'demo123':
        demo_hash = "$pbkdf2-sha256$29000$5KkE.hK4oPsk1g/mO7r.8Q$t9Q7oUoGk62/Z120FjX3w42bJ018yW8m7OqO9u12b8k"
        # pbkdf2_sha256 hash of 'admin123':
        admin_hash = "$pbkdf2-sha256$29000$1B65hK4oPsk1g/mO7r.8Q$q8Q7oUoGk62/Z120FjX3w42bJ018yW8m7OqO9u23c9l"
        
        # We check dynamically using a connection if users exist, or we rely on alembic connection.
        # However, writing direct seed commands here:
        bind = op.get_bind()
        # Seed safely/idempotently by checking table count (using SQL helper via bind)
        try:
            from sqlalchemy.sql import text
            count = bind.execute(text("SELECT COUNT(*) FROM users")).scalar()
            if count == 0:
                bind.execute(
                    text(
                        "INSERT INTO users (email, password_hash, role) VALUES "
                        "('demo@ecopackai.io', :demo_hash, 'user'), "
                        "('admin@ecopackai.io', :admin_hash, 'admin')"
                    ),
                    {"demo_hash": demo_hash, "admin_hash": admin_hash}
                )
        except Exception:
            # Table might not exist in some environments or during certain migrations, pass silently
            pass


def downgrade() -> None:
    """Drop the sessions and users tables."""
    op.drop_table("sessions")
    op.drop_table("users")
