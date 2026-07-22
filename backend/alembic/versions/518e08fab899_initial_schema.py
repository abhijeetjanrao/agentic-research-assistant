"""initial schema

Revision ID: 518e08fab899
Revises:
Create Date: 2026-07-16

Hand-written to match app.db.models exactly, since there's no live
database available yet to autogenerate against. Once a real MySQL
instance is running, future migrations should be generated with:

    alembic revision --autogenerate -m "description"

and reviewed before applying.
"""

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = "518e08fab899"
down_revision = None
branch_labels = None
depends_on = None


session_status_enum = sa.Enum("active", "completed", "failed", name="sessionstatus")
message_role_enum = sa.Enum("user", "assistant", "agent", name="messagerole")
document_status_enum = sa.Enum(
    "uploaded", "ingesting", "ingested", "failed", name="documentstatus"
)


def upgrade() -> None:
    op.create_table(
        "research_sessions",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("title", sa.String(length=255), nullable=False),
        sa.Column(
            "status",
            session_status_enum,
            nullable=False,
            server_default="active",
        ),
        sa.Column("created_at", sa.DateTime(), server_default=sa.func.now()),
        sa.Column(
            "updated_at",
            sa.DateTime(),
            server_default=sa.func.now(),
            onupdate=sa.func.now(),
        ),
    )

    op.create_table(
        "messages",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column(
            "session_id",
            sa.Integer(),
            sa.ForeignKey("research_sessions.id"),
            nullable=False,
        ),
        sa.Column("role", message_role_enum, nullable=False),
        sa.Column("agent_name", sa.String(length=64), nullable=True),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("created_at", sa.DateTime(), server_default=sa.func.now()),
    )
    op.create_index("ix_messages_session_id", "messages", ["session_id"])

    op.create_table(
        "documents",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column(
            "session_id",
            sa.Integer(),
            sa.ForeignKey("research_sessions.id"),
            nullable=False,
        ),
        sa.Column("filename", sa.String(length=255), nullable=False),
        sa.Column("file_path", sa.String(length=512), nullable=False),
        sa.Column(
            "status",
            document_status_enum,
            nullable=False,
            server_default="uploaded",
        ),
        sa.Column("num_chunks", sa.Integer(), nullable=True),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(), server_default=sa.func.now()),
    )
    op.create_index("ix_documents_session_id", "documents", ["session_id"])

    op.create_table(
        "reports",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column(
            "session_id",
            sa.Integer(),
            sa.ForeignKey("research_sessions.id"),
            nullable=False,
        ),
        sa.Column("title", sa.String(length=255), nullable=False),
        sa.Column("content_markdown", sa.Text(), nullable=False),
        sa.Column("citations_json", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(), server_default=sa.func.now()),
    )
    op.create_index("ix_reports_session_id", "reports", ["session_id"])


def downgrade() -> None:
    op.drop_table("reports")
    op.drop_table("documents")
    op.drop_table("messages")
    op.drop_table("research_sessions")
