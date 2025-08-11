"""add_document_chunks

Revision ID: 8d9176c7fa8d
Revises: 7d9176c7fa8c
Create Date: 2024-03-19 16:48:57.494597

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision: str = '8d9176c7fa8d'
down_revision: Union[str, None] = '7d9176c7fa8c'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

def upgrade() -> None:
    """Upgrade schema."""
    # Create ai_document_chunks table
    op.create_table(
        'ai_document_chunks',
        sa.Column('id', sa.String(), nullable=False),
        sa.Column('document_id', sa.String(), nullable=False),
        sa.Column('chunk_text', sa.Text(), nullable=False),
        sa.Column('page_number', sa.Integer(), nullable=True),
        sa.Column('chunk_index', sa.Integer(), nullable=True),
        sa.Column('embedding_id', sa.String(), nullable=True, unique=True),
        sa.Column('chapter_name', sa.String(), nullable=True),
        sa.Column('section_name', sa.String(), nullable=True),
        sa.Column('category', sa.String(), nullable=True),
        sa.Column('relevance_score', sa.Float(), nullable=True),
        sa.Column('last_accessed', sa.DateTime(), nullable=True),
        sa.PrimaryKeyConstraint('id'),
        sa.ForeignKeyConstraint(['document_id'], ['ai_documents.id'], ondelete='CASCADE'),
    )

def downgrade() -> None:
    """Downgrade schema."""
    # Drop table
    op.drop_table('ai_document_chunks') 