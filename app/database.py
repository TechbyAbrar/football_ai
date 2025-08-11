#database.py
from sqlalchemy import create_engine, text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Get database URL from environment variable
DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    raise ValueError("DATABASE_URL not found in environment variables")

# Create SQLAlchemy engine with optimized connection pooling
engine = create_engine(
    DATABASE_URL,
    pool_pre_ping=True,  # Enables automatic reconnection
    pool_size=20,  # Increased pool size for better performance
    max_overflow=30,  # Allow more overflow connections
    pool_timeout=30,  # Connection timeout in seconds
    pool_recycle=3600,  # Recycle connections every hour
    echo=False,  # Disabled in production for performance
)

# Create SessionLocal class
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Create Base class
Base = declarative_base()

def get_db():
    """Get database session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def check_user_auth(db, email: str) -> tuple[bool, bool]:
    """
    Check user authentication and staff status in account_userauth table.
    
    Args:
        db: Database session
        email: User email to check
        
    Returns:
        tuple: (exists, is_staff)
    """
    try:
        result = db.execute(
            text("SELECT email, is_staff FROM account_userauth WHERE email = :email"),
            {"email": email}
        ).first()
        
        exists = bool(result)
        is_staff = bool(result and result[1]) if exists else False
        
        return exists, is_staff
    except Exception as e:
        print(f"Error checking user auth: {e}")
        return False, False
    


def get_user_full_name(email: str):
    """
    Get user's full name from account_userauth table.
    
    Args:
        email: User email to lookup
        
    Returns:
        str: User's full name or None if not found
    """
    db = SessionLocal()
    try:
        result = db.execute(
            text("SELECT full_name FROM account_userauth WHERE email = :email AND is_verified = TRUE"),
            {"email": email}
        ).first()
        
        return result[0] if result else None
    except Exception as e:
        print(f"Error getting user full name: {e}")
        return None
    finally:
        db.close()
