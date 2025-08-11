"""initial migration

Revision ID: 8d9176c7fa8b
Revises: 729c04c2cec5
Create Date: 2025-07-19 10:40:43.224074

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '8d9176c7fa8b'
down_revision: Union[str, None] = '729c04c2cec5'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    # Rename tables to new naming convention
    op.rename_table('users', 'ai_users')
    op.rename_table('chat_sessions', 'ai_chat_sessions')
    op.rename_table('chat_messages', 'ai_chat_messages')
    op.rename_table('books', 'ai_documents')
    op.rename_table('book_chunks', 'ai_document_chunks')


def downgrade() -> None:
    """Downgrade schema."""
    # Revert table names to original
    op.rename_table('ai_users', 'users')
    op.rename_table('ai_chat_sessions', 'chat_sessions')
    op.rename_table('ai_chat_messages', 'chat_messages')
    op.rename_table('ai_documents', 'books')
    op.rename_table('ai_document_chunks', 'book_chunks')
