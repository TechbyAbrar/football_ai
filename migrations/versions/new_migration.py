"""remove_admin_add_category

Revision ID: 9d9176c7fa8c
Revises: 8d9176c7fa8b
Create Date: 2024-03-19 15:40:43.224074

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '9d9176c7fa8c'
down_revision: Union[str, None] = '8d9176c7fa8b'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    # Remove is_admin column from ai_users
    op.drop_column('ai_users', 'is_admin')
    
    # Add category column to ai_document_chunks
    op.add_column('ai_document_chunks', sa.Column('category', sa.String()))


def downgrade() -> None:
    """Downgrade schema."""
    # Add back is_admin column to ai_users
    op.add_column('ai_users', sa.Column('is_admin', sa.Boolean(), nullable=False, server_default='false'))
    
    # Remove category column from ai_document_chunks
    op.drop_column('ai_document_chunks', 'category') 