#api.py
from fastapi import FastAPI, HTTPException, Depends, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware
from sqlalchemy.orm import Session
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, EmailStr
from pydantic.config import ConfigDict
import logging
from datetime import datetime, timedelta, timezone
import os
from dotenv.main import load_dotenv

from app.models import Base, AI_User, AI_ChatSession, AI_ChatMessage, AI_Document
from app.ai_assistant import AIAssistant, AIAssistantError
from app.database import get_db, engine, check_user_auth

load_dotenv()

# Configure logging
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

# Create database tables
Base.metadata.create_all(bind=engine)

app = FastAPI(title="Football AI Assistant API")

# Configure CORS
origins = os.getenv("ALLOWED_ORIGINS", "http://localhost:3000,http://localhost:5173").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
    max_age=3600,
)

# Configure upload settings
# Maximum upload size: 500MB (handled at Uvicorn/ASGI level)
MAX_UPLOAD_SIZE = 500 * 1024 * 1024  # 500MB in bytes

app.max_upload_size = MAX_UPLOAD_SIZE

# Pydantic models
class BaseModelWithConfig(BaseModel):
    model_config = ConfigDict(from_attributes=True)

class EmailRequest(BaseModelWithConfig):
    email: str

class SessionCreateResponse(BaseModelWithConfig):
    session_id: str

class ChatRequest(BaseModelWithConfig):
    session_id: Optional[str] = None
    query_text: str
    email: str

class ChatResponseData(BaseModelWithConfig):
    response_id: str
    query_text: str
    response_text: str

class ChatResponse(BaseModelWithConfig):
    session_id: str
    data: List[ChatResponseData] = []

class SessionInfo(BaseModelWithConfig):
    title: str
    session_id: str
    session_start: datetime

class SessionsByDateResponse(BaseModelWithConfig):
    today: List[SessionInfo] = []
    yesterday: List[SessionInfo] = []
    last_week: List[SessionInfo] = []
    last_month: List[SessionInfo] = []
    last_year: List[SessionInfo] = []

class DocumentInfo(BaseModelWithConfig):
    email: str
    document_id: str
    document_name: str
    upload_on: datetime
    author: Optional[str] = None
    category: Optional[str] = None
    is_public: bool = True

class DocumentListResponse(BaseModelWithConfig):
    documents: List[DocumentInfo] = []

class DeleteDocumentsRequest(BaseModelWithConfig):
    email: str
    document_ids: List[str]

# Configure multipart upload size
@app.on_event("startup")
async def configure_upload_limits():
    """Configure larger upload limits on startup."""
    try:
        # Set Starlette's multipart form parser to accept larger files
        import starlette.datastructures
        # The max_size is handled at the ASGI level via Uvicorn
        # Additional configuration can be set here if needed
        logger.info(f"Upload limit configured to {MAX_UPLOAD_SIZE / (1024*1024):.0f}MB")
    except Exception as e:
        logger.error(f"Error configuring upload limits: {e}")

@app.post("/ai/chat_session/")
async def create_chat_session(
    request: EmailRequest,
    db: Session = Depends(get_db)
) -> SessionCreateResponse:
    """Create a new chat session for a user."""
    try:
        # Check if user exists in auth database
        exists, _ = check_user_auth(db, request.email)
        if not exists:
            raise HTTPException(
                status_code=403,
                detail="User not found in authentication records"
            )

        # Check if user exists in ai_users, if not create
        user = db.query(AI_User).filter(AI_User.email == request.email).first()
        if not user:
            user = AI_User(email=request.email)
            db.add(user)
            db.commit()
            db.refresh(user)
            logger.info(f"Created new user with email: {request.email}")

        # Create new chat session
        session = AI_ChatSession(
            email=request.email,
            title="New Consultation",
            created_at=datetime.now(timezone.utc)
        )
        db.add(session)
        db.commit()
        db.refresh(session)
        
        logger.info(f"Created new session with ID: {session.id}")
        return SessionCreateResponse(session_id=str(session.id))

    except Exception as e:
        logger.error(f"Error creating chat session: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to create session: {str(e)}"
        )


@app.post("/ai/chat/", response_model=ChatResponse)
async def chat(request: ChatRequest, db: Session = Depends(get_db)):
    """Process a user message and return the AI's response with context.
    
    Authentication flow:
    1. First check if email exists in account_userauth
    2. If not in account_userauth, deny access
    3. If in account_userauth, check if exists in ai_users
    4. If not in ai_users, create new ai_user
    5. Create/verify session and process message
    """
    try:
        # Input validation
        if not request.query_text.strip():
            raise HTTPException(
                status_code=400,
                detail={"status": "error", "message": "Query text cannot be empty"}
            )

        # Step 1: Check if user exists in account_userauth
        exists, is_subscribed = check_user_auth(db, request.email)
        if not exists or not is_subscribed:
            raise HTTPException(
                status_code=403,
                detail={"status": "error", "message": "Access denied. User not authorized."}
            )

        # Step 2: Check if user exists in ai_users
        user = db.query(AI_User).filter(AI_User.email == request.email).first()
        
        # Step 3: If user exists in account_userauth but not in ai_users, create new ai_user
        if not user:
            user = AI_User(email=request.email)
            db.add(user)
            db.commit()
            db.refresh(user)
            logger.info(f"Created new AI user with email: {request.email}")

        # Handle session management
        session_id = request.session_id
        if not session_id:
            # Create new chat session
            initial_title = (
                request.query_text[:47] + "..."
                if len(request.query_text) > 50
                else request.query_text
            )
            
            session = AI_ChatSession(
                email=request.email,
                title=initial_title,
                created_at=datetime.utcnow()
            )
            db.add(session)
            db.commit()
            db.refresh(session)
            session_id = str(session.id)
        else:
            # Verify session exists and belongs to user
            session = db.query(AI_ChatSession).filter(
                AI_ChatSession.id == session_id,
                AI_ChatSession.email == request.email
            ).first()
            if not session:
                raise HTTPException(
                    status_code=404,
                    detail={"status": "error", "message": "Session not found or unauthorized"}
                )

        # Process message with AI
        try:
            ai_assistant = AIAssistant(db)
            response_text = ai_assistant.process_message(
                session_id=session_id,
                query_text=request.query_text,
                email=request.email
            )
            
            # Store message
            message = AI_ChatMessage(
                session_id=session_id,
                query_text=request.query_text,
                response_text=response_text,
                timestamp=datetime.utcnow()
            )
            db.add(message)
            
            # Update session
            session.last_updated = datetime.utcnow()
            if len(request.query_text) <= 50 and session.title == "New Consultation":
                session.title = request.query_text
            
            db.commit()
            db.refresh(message)
            
            return ChatResponse(
                session_id=session_id,
                data=[
                    ChatResponseData(
                        response_id=str(message.id),
                        query_text=message.query_text,
                        response_text=message.response_text
                    )
                ]
            )
            
        except AIAssistantError as e:
            raise HTTPException(
                status_code=500,
                detail={"status": "error", "message": "Failed to process message", "error": str(e)}
            )
            
    except Exception as e:
        logger.error(f"Error in chat endpoint: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail={"status": "error", "message": "Failed to process request", "error": str(e)}
        )


@app.get("/ai/all_chat/")
async def get_all_chat(
    session_id: str,
    db: Session = Depends(get_db)
) -> ChatResponse:
    """
    Get all chat messages for a specific session.
    Args:
        session_id: The session ID to retrieve messages for
        db: Database session
    Returns:
        ChatResponse containing all messages in the session
    """

    try:

        # Validate session exists

        session = (
            db.query(AI_ChatSession).filter(AI_ChatSession.id == session_id).first()
        )

        if not session:

            raise HTTPException(status_code=404, detail="Session not found")

        # Get all messages for the session

        messages = (
            db.query(AI_ChatMessage)
            .filter(AI_ChatMessage.session_id == session_id)
            .order_by(AI_ChatMessage.timestamp.asc())
            .all()
        )

        data = [
            ChatResponseData(
                response_id=str(msg.id),
                query_text=msg.query_text,
                response_text=msg.response_text,
            )
            for msg in messages
        ]

        return ChatResponse(session_id=session_id, data=data)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting all chat: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")


@app.get("/ai/all_sessions/")
async def get_all_sessions(
    email: str,
    db: Session = Depends(get_db)
) -> SessionsByDateResponse:
    """
    Get all sessions for a specific user, grouped by date.
    Args:
        email: Email address to get sessions for
        db: Database session
    Returns:

        SessionsByDateResponse with sessions grouped by date
    """
    try:
        # Get current time for date calculations

        now = datetime.utcnow()

        today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
        yesterday_start = today_start - timedelta(days=1)
        week_start = today_start - timedelta(days=7)
        month_start = today_start - timedelta(days=30)
        year_start = today_start - timedelta(days=365)

        # Query all sessions for the user

        sessions = (
            db.query(AI_ChatSession)
            .filter(AI_ChatSession.email == email)
            .order_by(AI_ChatSession.created_at.desc())
            .all()
        )

        # Group sessions by date

        grouped_sessions = {
            "today": [],
            "yesterday": [],
            "last_week": [],
            "last_month": [],
            "last_year": [],
        }

        for session in sessions:

            session_info = SessionInfo(
                title=session.title,
                session_id=str(session.id),
                session_start=session.created_at,
            )

            if session.created_at >= today_start:
                grouped_sessions["today"].append(session_info)

            elif session.created_at >= yesterday_start:
                grouped_sessions["yesterday"].append(session_info)

            elif session.created_at >= week_start:
                grouped_sessions["last_week"].append(session_info)

            elif session.created_at >= month_start:
                grouped_sessions["last_month"].append(session_info)

            elif session.created_at >= year_start:
                grouped_sessions["last_year"].append(session_info)

        return SessionsByDateResponse(**grouped_sessions)

    except Exception as e:
        logger.error(f"Error getting all sessions: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")


@app.post("/ai/upload/")
async def upload_document(
    email: str = Form(...),
    files: List[UploadFile] = File(...),
    author: Optional[str] = Form(None),
    description: Optional[str] = Form(None),
    category: Optional[str] = Form(None),
    is_public: bool = Form(True),
    db: Session = Depends(get_db)
) -> Dict[str, Any]:
    """Upload and process multiple documents. Only accessible by staff users."""
    try:
        # Log upload initiation
        logger.info(f"Upload started for {email}: {len(files)} file(s)")
        
        # Check if user exists and is staff
        exists, is_subscribed = check_user_auth(db, email)
        if not exists:
            raise HTTPException(
                status_code=403,
                detail="User not found in authentication records"
            )
        if not is_subscribed:
            raise HTTPException(
                status_code=403,
                detail="Only subscribed members can upload documents"
            )

        # Validate category if provided
        valid_categories = [
            "Nutrition", 
            "Strength and Conditioning", 
            "Training Program Schedules", 
            "General Player Advice", 
            "Injury Prevention and Management", 
            "Mental Well-Being",
            "Performance Analytics Tips", 
            "Tactical Development",
            "Uncategorized"  # Allow explicit uncategorized
        ]
        if category and category not in valid_categories:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid category. Must be one of: {', '.join(valid_categories)}"
            )

        uploaded_docs = []
        ai_assistant = AIAssistant(db)

        for file in files:
            # Validate file
            if not file.filename.lower().endswith('.pdf'):
                logger.warning(f"Skipping non-PDF file: {file.filename}")
                continue

            try:
                # Log file size and processing start
                file_size = file.size if hasattr(file, 'size') else "unknown"
                logger.info(f"Processing file: {file.filename} (size: {file_size} bytes)")
                
                # Upload and process document with category
                document_id = await ai_assistant.upload_and_process_document(
                    email=email,
                    file=file,  # Pass the file directly, filename will be taken from it
                    category=category
                )

                # Update other document metadata
                document = db.query(AI_Document).filter(AI_Document.id == document_id).first()
                if document:
                    document.author = author
                    document.description = description
                    document.is_public = is_public
                    db.commit()

                logger.info(f"Successfully processed file: {file.filename} (id: {document_id})")
                uploaded_docs.append({
                    "filename": file.filename,  # Use original filename from the uploaded file
                    "document_id": document_id,
                    "category": category or "Uncategorized",
                    "status": "success"
                })

            except Exception as e:
                logger.error(f"Error processing {file.filename}: {str(e)}")
                uploaded_docs.append({
                    "filename": file.filename,
                    "status": "error",
                    "error": str(e)
                })

        return {
            "status": "completed",
            "message": f"Processed {len(uploaded_docs)} documents",
            "results": uploaded_docs
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Upload error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/ai/document_list/", response_model=DocumentListResponse)
async def get_document_list(email: EmailStr, db: Session = Depends(get_db)):
    """Get list of all documents. Only accessible by staff users."""
    try:
        # Check if user exists and is staff
        exists, is_subscribed = check_user_auth(db, email)
        if not exists:
            raise HTTPException(
                status_code=403,
                detail="User not found in authentication records"
            )
        if not is_subscribed:
            raise HTTPException(
                status_code=403,
                detail="Only subscribed members can access the document list"
            )

        # Staff can see all documents
        documents = db.query(AI_Document).all()

        document_list = []
        for document in documents:
            document_list.append(
                DocumentInfo(
                    email=document.admin_email,
                    document_id=document.id,
                    document_name=document.document_name,
                    upload_on=document.upload_on,
                    author=document.author,
                    category=document.category,
                    is_public=document.is_public,
                )
            )

        return DocumentListResponse(documents=document_list)

    except Exception as e:
        logger.error(f"Error getting document list: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get document list: {str(e)}"
        )


@app.delete("/ai/delete_documents/")
async def delete_documents(
    request: DeleteDocumentsRequest,
    db: Session = Depends(get_db)
):
    """Delete documents and their associated data. Only accessible by staff users."""
    try:
        # Check if user exists and is staff
        exists, is_subscribed = check_user_auth(db, request.email)
        if not exists:
            raise HTTPException(
                status_code=403,
                detail="User not found in authentication records"
            )
        if not is_subscribed:
            raise HTTPException(
                status_code=403,
                detail="Only subscribed members can delete documents"
            )

        # Initialize AI Assistant
        ai_assistant = AIAssistant(db)

        # Delete documents and update knowledge base
        try:
            deleted_count = ai_assistant.delete_documents(
                request.email,
                request.document_ids
            )

            return {
                "success": True,
                "message": f"Successfully deleted {deleted_count} documents",
                "deleted_count": deleted_count
            }

        except AIAssistantError as e:
            logger.error(f"Error deleting documents: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in delete_documents: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail={
                "success": False,
                "message": "Failed to delete documents",
                "error": str(e)
            }
        )


@app.get("/ai/search/")
async def search_sessions(q: str, email: EmailStr, db: Session = Depends(get_db)):
    """
    Search sessions by title for a specific user.
    Args:
        q: Search query string
        email: User's email address
        db: Database session
    Returns:
        List of matching sessions
    """
    try:

        sessions = (
            db.query(AI_ChatSession)
            .filter(AI_ChatSession.email == email, AI_ChatSession.title.ilike(f"%{q}%"))
            .order_by(AI_ChatSession.created_at.desc())
            .all()
        )

        results = [
            {
                "title": session.title,
                "session_id": str(session.id),
                "session_start": session.created_at.isoformat() + "Z",
            }
            for session in sessions
        ]

        return {"results": results}

    except Exception as e:

        logger.error(f"Error searching sessions: {str(e)}")

        raise HTTPException(status_code=500, detail="Internal server error")

if __name__ == "__main__":

    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)



# abraham
class DeleteSessionChatsRequest(BaseModel):
    email: EmailStr
    session_id: str

@app.delete("/ai/delete_session_chats/")
async def delete_session_chats(
    request: DeleteSessionChatsRequest,
    db: Session = Depends(get_db)
):
    """
    Delete all chat messages from a specific session.
    """
    try:
        # 1️⃣ Check if user exists
        exists, is_subscribed = check_user_auth(db, request.email)
        if not exists or not is_subscribed:
            raise HTTPException(
                status_code=403,
                detail="User not found"
            )

        # 2️⃣ Check if the session belongs to the user
        session = db.query(AI_ChatSession).filter(
            AI_ChatSession.id == request.session_id,
            AI_ChatSession.email == request.email
        ).first()
        if not session:
            raise HTTPException(
                status_code=404,
                detail="Session not found or unauthorized"
            )

        # 3️⃣ Delete all messages in the session
        deleted_count = db.query(AI_ChatMessage).filter(
            AI_ChatMessage.session_id == request.session_id
        ).delete()
        
        # 4️⃣ Delete the session itself
        db.delete(session)
        db.commit()

        return {
            "success": True,
            "message": f"Deleted session {request.session_id} with {deleted_count} messages"
        }

    except HTTPException:
        db.rollback()
        raise
    except Exception as e:
        db.rollback()
        logger.error(f"Error deleting session chats: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to delete session chats: {str(e)}"
        )