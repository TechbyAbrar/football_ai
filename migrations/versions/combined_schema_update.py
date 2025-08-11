"""combined_schema_update

Revision ID: 9d9176c7fa8e
Revises: 8d9176c7fa8b
Create Date: 2024-03-19 17:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.engine.reflection import Inspector

# revision identifiers, used by Alembic.
revision: str = '9d9176c7fa8e'
down_revision: Union[str, None] = '8d9176c7fa8b'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

def upgrade() -> None:
    """Upgrade schema."""
    # Check if is_admin column exists before trying to drop it
    conn = op.get_bind()
    inspector = Inspector.from_engine(conn)
    columns = [col['name'] for col in inspector.get_columns('ai_users')]
    if 'is_admin' in columns:
        op.drop_column('ai_users', 'is_admin')
    
    # Create ai_documents table if it doesn't exist
    if not inspector.has_table('ai_documents'):
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
        
        # Create indexes for ai_documents
        op.create_index(op.f('ix_ai_documents_admin_email'), 'ai_documents', ['admin_email'], unique=False)
        op.create_index(op.f('ix_ai_documents_upload_on'), 'ai_documents', ['upload_on'], unique=False)
    
    # Create ai_document_chunks table if it doesn't exist
    if not inspector.has_table('ai_document_chunks'):
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
    # Drop tables if they exist
    conn = op.get_bind()
    inspector = Inspector.from_engine(conn)
    
    if inspector.has_table('ai_document_chunks'):
        op.drop_table('ai_document_chunks')
    
    if inspector.has_table('ai_documents'):
        op.drop_index(op.f('ix_ai_documents_upload_on'), table_name='ai_documents')
        op.drop_index(op.f('ix_ai_documents_admin_email'), table_name='ai_documents')
        op.drop_table('ai_documents')
    
    # Add back is_admin column if it doesn't exist
    columns = [col['name'] for col in inspector.get_columns('ai_users')]
    if 'is_admin' not in columns:
        op.add_column('ai_users', sa.Column('is_admin', sa.Boolean(), nullable=True, server_default='false')) 