"""add_document_columns

Revision ID: 7d9176c7fa8c
Revises: 6d5037877ca0
Create Date: 2024-03-19 16:47:57.494597

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = '7d9176c7fa8c'
down_revision: Union[str, None] = '6d5037877ca0'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

def upgrade() -> None:
    """Upgrade schema."""
    # Create ai_documents table with all required columns
    op.create_table(
        'ai_documents',
        sa.Column('id', sa.String(), nullable=False),
        sa.Column('admin_email', sa.String(), nullable=False),
        sa.Column('document_name', sa.String(), nullable=False),
        sa.Column('file_path', sa.String(), nullable=False),
        sa.Column('upload_on', sa.DateTime(), nullable=False, server_default=sa.text('now()')),
        sa.Column('processed', sa.Boolean(), nullable=False, server_default='false'),
        sa.Column('page_count', sa.Integer(), nullable=False, server_default='0'),
        sa.Column('is_public', sa.Boolean(), nullable=False, server_default='true'),
        sa.Column('author', sa.String(), nullable=True),
        sa.Column('description', sa.Text(), nullable=True),
        sa.Column('category', sa.String(), nullable=True),
        sa.PrimaryKeyConstraint('id'),
        sa.ForeignKeyConstraint(['admin_email'], ['ai_users.email'], ondelete='CASCADE'),
    )
    
    # Create indexes
    op.create_index(op.f('ix_ai_documents_admin_email'), 'ai_documents', ['admin_email'], unique=False)
    op.create_index(op.f('ix_ai_documents_upload_on'), 'ai_documents', ['upload_on'], unique=False)

def downgrade() -> None:
    """Downgrade schema."""
    # Drop indexes
    op.drop_index(op.f('ix_ai_documents_upload_on'), table_name='ai_documents')
    op.drop_index(op.f('ix_ai_documents_admin_email'), table_name='ai_documents')
    
    # Drop table
    op.drop_table('ai_documents') 