import os
import logging
import uuid
import asyncio
from typing import List, Dict, Any, Optional, Tuple, cast, TypeVar, Union
from datetime import datetime
import tempfile
import shutil

from fastapi import UploadFile
from sqlalchemy.orm import Session
from openai._client import OpenAI
from sentence_transformers.SentenceTransformer import SentenceTransformer
import chromadb
from chromadb.config import Settings
from chromadb.api.models.Collection import Collection
from PyPDF2._reader import PdfReader
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
import re
from dotenv.main import load_dotenv
from app.database import get_user_full_name
from app.models import AI_User, AI_ChatSession, AI_ChatMessage, AI_Document, AI_DocumentChunk

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
    nltk.download('punkt_tab')
except LookupError:
    nltk.download('punkt')
    

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

class AIAssistantError(Exception):
    """Custom exception for AI Assistant errors"""
    pass

class AIAssistant:
    def __init__(self, db_session: Session) -> None:
        """Initialize the AI Assistant."""
        self.db = db_session

        
        # Initialize OpenAI client with validation
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        print(self.openai_api_key)
        if not self.openai_api_key:
            raise AIAssistantError("OpenAI API key not found in environment variables")
        
        # Validate OpenAI API key format
        if not self.openai_api_key.startswith('sk-'):
            raise AIAssistantError("Invalid OpenAI API key format. Key should start with 'sk-'")
            
        try:
            self.openai_client = OpenAI(api_key=self.openai_api_key)
            # Test the API key with a minimal request
            self.openai_client.models.list()
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI client: {e}")
            raise AIAssistantError(f"Failed to initialize OpenAI client. Please check your API key: {e}")
        
        # Initialize sentence transformer for embeddings
        try:
            self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            raise AIAssistantError(f"Failed to initialize embedding model: {e}")
        
        # Initialize ChromaDB client
        try:
            self.chroma_client = chromadb.PersistentClient(
                path="./chroma_db",
                settings=Settings(anonymized_telemetry=False)
            )
        except Exception as e:
            logger.error(f"Failed to initialize ChromaDB: {e}")
            raise AIAssistantError(f"Failed to initialize vector database: {e}")
        
        # Directory for storing uploaded PDFs
        self.upload_dir = "uploads"
        os.makedirs(self.upload_dir, exist_ok=True)

    def get_or_create_collection(self, email: str) -> Collection:
        """Get or create ChromaDB collections for a user.
        Returns the public collection that contains all public documents."""
        try:
            # Get or create public collection for all users
            public_collection = self.chroma_client.get_or_create_collection(
                name="public_documents",
                metadata={"type": "public"}
            )
            return cast(Collection, public_collection)
        except Exception as e:
            logger.error(f"Error getting/creating collection: {e}")
            raise AIAssistantError(f"Failed to get/create collection: {e}")

    async def upload_and_process_document(self, email: str, file: UploadFile, category: Optional[str] = None) -> str:
        """Upload and process a document."""
        file_path = ""
        try:
            # Get original filename from the uploaded file
            filename = file.filename
            
            # Generate unique document ID and file path
            document_id = str(uuid.uuid4())
            file_path = os.path.join(self.upload_dir, f"{document_id}_{filename}")
            
            # Save the uploaded file
            with open(file_path, "wb") as buffer:
                content = await file.read()
                buffer.write(content)
            
            # Extract text from PDF
            text_content, page_count = self._extract_pdf_text(file_path)
            
            # Create document record in database with the provided category
            document = AI_Document(
                id=document_id,
                admin_email=email,
                document_name=filename,  # Use original filename
                file_path=file_path,
                page_count=page_count,
                processed=False,
                is_public=True,
                category=category or "Uncategorized"  # Use provided category or default
            )
            self.db.add(document)
            self.db.commit()
            self.db.refresh(document)
            
            # Process text into chunks and store in vector database
            await self._process_and_store_chunks(document_id, email, text_content, filename, category)
            
            # Mark document as processed
            document.processed = True
            self.db.commit()
            
            logger.info(f"Successfully processed document {filename} for admin {email}")
            return document_id
            
        except Exception as e:
            logger.error(f"Error processing document {file.filename}: {e}")
            # Clean up on error
            if file_path and os.path.exists(file_path):
                try:
                    if os.path.isfile(file_path):
                        os.unlink(file_path)
                except OSError as ose:
                    logger.error(f"Error removing file {file_path}: {ose}")
            self.db.rollback()
            raise AIAssistantError(f"Failed to process document: {e}")

    def _clean_text(self, text: str) -> str:
        """Clean text by removing null characters and other problematic characters."""
        try:
            # Remove null characters
            text = text.replace('\x00', '')
            
            # Remove other potential problematic characters
            text = text.replace('\r', ' ')
            text = text.replace('\t', ' ')
            
            # Normalize whitespace
            text = ' '.join(text.split())
            
            # Remove any remaining control characters except newlines
            text = ''.join(char for char in text if char == '\n' or char >= ' ')
            
            return text
        except Exception as e:
            logger.error(f"Error cleaning text: {e}")
            return text

    def _extract_pdf_text(self, file_path: str) -> tuple[str, int]:
        """
        Extract text content from a PDF file.
        
        Args:
            file_path: Path to the PDF file
            
        Returns:
            tuple: (extracted_text, page_count)
        """
        try:
            text_content = ""
            page_count = 0
            
            with open(file_path, 'rb') as file:
                pdf_reader = PdfReader(file)
                page_count = len(pdf_reader.pages)
                
                for page_num, page in enumerate(pdf_reader.pages, 1):
                    try:
                        page_text = page.extract_text()
                        if page_text and page_text.strip():
                            # Clean the text before adding page markers
                            cleaned_text = self._clean_text(page_text)
                            if cleaned_text.strip():
                                text_content += f"\n[Page {page_num}]\n{cleaned_text}\n"
                    except Exception as e:
                        logger.warning(f"Error extracting text from page {page_num}: {e}")
                        continue
            
            # Final cleaning of the entire text
            text_content = self._clean_text(text_content)
            
            if not text_content.strip():
                raise AIAssistantError("No text could be extracted from the PDF")
            
            return text_content, page_count
            
        except Exception as e:
            logger.error(f"Error extracting PDF text: {e}")
            raise AIAssistantError(f"Failed to extract text from PDF: {e}")

    async def _process_and_store_chunks(self, document_id: str, email: str, text_content: str, document_name: str, category: Optional[str] = None):
        """
        Process text content into chunks and store in vector database.
        """
        try:
            # Get public collection
            collection = self.get_or_create_collection(email)
            
            # Split text into chunks with category-aware processing
            chunks = self._split_text_into_chunks(text_content)
            
            # Use provided category or default
            chunk_category = category or "Uncategorized"
            
            # Process chunks in batches for better performance
            batch_size = 50
            for i in range(0, len(chunks), batch_size):
                batch_chunks = chunks[i:i + batch_size]
                
                # Prepare batch data
                chunk_embeddings = []
                chunk_texts = []
                chunk_metadata = []
                chunk_ids = []
                
                for j, chunk_text in enumerate(batch_chunks, i):
                    try:
                        # Clean the chunk text
                        chunk_text = self._clean_text(chunk_text)
                        if not chunk_text.strip():
                            continue
                        
                        # Generate unique ID for chunk
                        chunk_embedding_id = f"{document_id}_chunk_{j}"
                        
                        # Create metadata - ensure all values are strings
                        metadata = {
                            "document_id": str(document_id),
                            "document_name": str(document_name),
                            "chunk_index": str(j),
                            "category": str(chunk_category),  # Use the same category for all chunks
                            "is_public": "true"  # Mark all documents as public
                        }
                        
                        # Get embedding
                        embedding = self.embedding_model.encode(chunk_text).tolist()
                        
                        # Append to batch lists
                        chunk_embeddings.append(embedding)
                        chunk_texts.append(chunk_text)
                        chunk_metadata.append(metadata)
                        chunk_ids.append(chunk_embedding_id)
                        
                        # Create database record with the same category
                        chunk = AI_DocumentChunk(
                            document_id=document_id,
                            chunk_text=chunk_text,
                            chunk_index=j,
                            embedding_id=chunk_embedding_id,
                            category=chunk_category  # Use the same category for all chunks
                        )
                        self.db.add(chunk)
                    except Exception as e:
                        logger.warning(f"Error processing chunk {j}: {e}")
                        continue
                
                if chunk_embeddings:  # Only add to ChromaDB if we have valid chunks
                    # Batch insert into ChromaDB
                    collection.add(
                        ids=chunk_ids,
                        embeddings=chunk_embeddings,
                        documents=chunk_texts,
                        metadatas=chunk_metadata
                    )
                    
                    # Commit database changes
                    self.db.commit()
                
            logger.info(f"Stored {len(chunks)} chunks for document {document_name}")
            
        except Exception as e:
            logger.error(f"Error processing chunks: {e}")
            self.db.rollback()
            raise AIAssistantError(f"Failed to process chunks: {e}")

    def _split_text_into_chunks(self, text_content: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
        """
        Split text content into overlapping chunks.
        
        Args:
            text_content: The full text content
            chunk_size: Maximum size of each chunk in characters
            overlap: Number of characters to overlap between chunks
            
        Returns:
            List of chunk text strings
        """
        chunks = []
        current_page = 1
        
        # Split by pages first
        page_sections = re.split(r'\[Page (\d+)\]', text_content)
        
        for i in range(1, len(page_sections), 2):
            if i + 1 < len(page_sections):
                page_num = int(page_sections[i])
                page_text = page_sections[i + 1].strip()
                
                if not page_text:
                    continue
                
                # Split page text into sentences for better chunk boundaries
                sentences = sent_tokenize(page_text)
                
                current_chunk = ""
                for sentence in sentences:
                    # Check if adding this sentence would exceed chunk size
                    if len(current_chunk) + len(sentence) > chunk_size and current_chunk:
                        # Save current chunk
                        chunks.append(current_chunk.strip())
                        
                        # Start new chunk with overlap
                        if overlap > 0:
                            words = current_chunk.split()
                            overlap_words = words[-overlap//10:] if len(words) > overlap//10 else words
                            current_chunk = " ".join(overlap_words) + " " + sentence
                        else:
                            current_chunk = sentence
                    else:
                        current_chunk += " " + sentence
                
                # Add remaining text as final chunk for this page
                if current_chunk.strip():
                    chunks.append(current_chunk.strip())
        
        # If no page markers found, split the entire text
        if not chunks:
            sentences = sent_tokenize(text_content)
            current_chunk = ""
            
            for sentence in sentences:
                if len(current_chunk) + len(sentence) > chunk_size and current_chunk:
                    chunks.append(current_chunk.strip())
                    
                    if overlap > 0:
                        words = current_chunk.split()
                        overlap_words = words[-overlap//10:] if len(words) > overlap//10 else words
                        current_chunk = " ".join(overlap_words) + " " + sentence
                    else:
                        current_chunk = sentence
                else:
                    current_chunk += " " + sentence
            
            if current_chunk.strip():
                chunks.append(current_chunk.strip())
        
        return chunks
    
    def _build_context_from_chunks(self, chunks: List[Dict]) -> tuple[str, List[Dict]]:
        """Build context text from retrieved chunks."""
        documents_context = {}
        references = []
        
        for chunk in chunks:
            document_name = chunk['metadata']['document_name']
            if document_name not in documents_context:
                documents_context[document_name] = []
            documents_context[document_name].append(chunk)
            
            references.append({
                'document_name': document_name,
                'page_number': chunk['metadata'].get('page_number'),
                'chunk_text': chunk['document'],
                'relevance_score': chunk['distance']
            })
        
        context_text = ""
        if documents_context:
            context_text = "\n\nRelevant information from your uploaded documents:\n"
            
            for document_name, document_chunks in documents_context.items():
                context_text += f"\nFrom document '{document_name}':\n"
                
                for chunk in document_chunks:
                    page_info = f" (Page {chunk['metadata'].get('page_number')})" if chunk['metadata'].get('page_number') else ""
                    context_text += f"{chunk['document']}{page_info}\n"
        
        # Sort references by relevance score
        references.sort(key=lambda x: x['relevance_score'], reverse=True)
        
        return context_text, references
    
    def reprocess_existing_chunks_to_chroma(self) -> Dict[str, Any]:
        """
        Reprocess existing document chunks from PostgreSQL into ChromaDB.
        This method will:
        1. Read all chunks from the database
        2. Generate embeddings
        3. Store them in the public ChromaDB collection
        """
        try:
            # Ensure any existing transaction is rolled back
            self.db.rollback()
            
            # Get the public collection
            collection = self.get_or_create_collection("")  # Empty email as we're using public collection
            
            # Get all documents and their chunks
            documents = self.db.query(AI_Document).all()
            
            total_chunks = 0
            processed_chunks = 0
            failed_chunks = 0
            
            # Process chunks in batches
            batch_size = 50
            
            for document in documents:
                try:
                    logger.info(f"Processing chunks for document: {document.document_name}")
                    
                    # Get all chunks for this document
                    chunks = (self.db.query(AI_DocumentChunk)
                            .filter(AI_DocumentChunk.document_id == document.id)
                            .order_by(AI_DocumentChunk.chunk_index)
                            .all())
                    
                    total_chunks += len(chunks)
                    
                    # Process chunks in batches
                    for i in range(0, len(chunks), batch_size):
                        try:
                            batch_chunks = chunks[i:i + batch_size]
                            
                            # Prepare batch data
                            chunk_embeddings = []
                            chunk_texts = []
                            chunk_metadata = []
                            chunk_ids = []
                            
                            for chunk in batch_chunks:
                                try:
                                    # Clean the chunk text
                                    chunk_text = self._clean_text(chunk.chunk_text)
                                    if not chunk_text.strip():
                                        continue
                                    
                                    # Generate embedding
                                    embedding = self.embedding_model.encode(chunk_text).tolist()
                                    
                                    # Create metadata
                                    metadata = {
                                        "document_id": str(document.id),
                                        "document_name": str(document.document_name),
                                        "chunk_index": str(chunk.chunk_index),
                                        "category": str(document.category or "Uncategorized"),
                                        "is_public": "true",
                                        "page_number": str(chunk.page_number) if chunk.page_number else "",
                                        "chapter_name": str(chunk.chapter_name) if chunk.chapter_name else "",
                                        "section_name": str(chunk.section_name) if chunk.section_name else ""
                                    }
                                    
                                    # Use existing embedding_id or generate new one
                                    chunk_id = chunk.embedding_id or f"{document.id}_chunk_{chunk.chunk_index}"
                                    
                                    # Update the chunk's embedding_id if it didn't have one
                                    if not chunk.embedding_id:
                                        chunk.embedding_id = chunk_id
                                    
                                    # Append to batch lists
                                    chunk_embeddings.append(embedding)
                                    chunk_texts.append(chunk_text)
                                    chunk_metadata.append(metadata)
                                    chunk_ids.append(chunk_id)
                                    
                                    processed_chunks += 1
                                    
                                except Exception as e:
                                    logger.error(f"Error processing chunk {chunk.id}: {e}")
                                    failed_chunks += 1
                                    continue
                            
                            if chunk_embeddings:  # Only add to ChromaDB if we have valid chunks
                                try:
                                    # Start a new transaction for this batch
                                    self.db.begin_nested()
                                    
                                    # Batch insert into ChromaDB
                                    collection.add(
                                        ids=chunk_ids,
                                        embeddings=chunk_embeddings,
                                        documents=chunk_texts,
                                        metadatas=chunk_metadata
                                    )
                                    
                                    # Commit this batch
                                    self.db.commit()
                                    
                                except Exception as e:
                                    logger.error(f"Error adding batch to ChromaDB: {e}")
                                    self.db.rollback()
                                    failed_chunks += len(chunk_embeddings)
                                    processed_chunks -= len(chunk_embeddings)
                                    continue
                                
                        except Exception as e:
                            logger.error(f"Error processing batch: {e}")
                            self.db.rollback()
                            continue
                    
                    logger.info(f"Completed processing document: {document.document_name}")
                    
                except Exception as e:
                    logger.error(f"Error processing document {document.document_name}: {e}")
                    self.db.rollback()
                    continue
            
            return {
                "status": "success",
                "total_chunks": total_chunks,
                "processed_chunks": processed_chunks,
                "failed_chunks": failed_chunks,
                "message": f"Successfully processed {processed_chunks} chunks, {failed_chunks} failed"
            }
            
        except Exception as e:
            self.db.rollback()
            logger.error(f"Error reprocessing chunks: {e}")
            raise AIAssistantError(f"Failed to reprocess chunks: {e}")

    def quick_reset_and_reprocess(self) -> Dict[str, Any]:
        """
        Quick reset of ChromaDB and reprocess all existing chunks.
        This is a convenience method that combines reset and reprocessing.
        """
        try:
            # Ensure any existing transaction is rolled back
            self.db.rollback()
            
            # First reset the ChromaDB
            self.reset_user_collection("")  # Empty email as we're using public collection
            
            # Then reprocess all chunks
            result = self.reprocess_existing_chunks_to_chroma()
            
            return {
                "status": "success",
                "reset": "completed",
                "reprocess": result
            }
            
        except Exception as e:
            self.db.rollback()
            logger.error(f"Error in quick reset and reprocess: {e}")
            raise AIAssistantError(f"Failed to reset and reprocess: {e}")
    
    def _retrieve_relevant_context(self, email: str, query: str, n_results: int = 10) -> List[Dict]:
        """
        Retrieve relevant context from all public documents.
        """
        try:
            # Get public collection
            collection = self.get_or_create_collection(email)
            
            # Get all public documents
            documents = self.db.query(AI_Document).filter(AI_Document.is_public == True).all()
            
            # If no documents exist, return empty list
            if not documents:
                return []
            
            # Get query embedding
            query_embedding = self.embedding_model.encode(query).tolist()
            
            # Search for similar chunks
            results = collection.query(
                query_embeddings=[query_embedding],
                n_results=min(n_results, len(documents)),  # Don't request more results than documents
                include=['documents', 'metadatas', 'distances']
            )
            
            relevant_chunks = []
            
            if results and results.get('ids') and results['ids'][0]:  # Check if we have any results
                for i in range(len(results['ids'][0])):  # Access first query's results
                    try:
                        document_id = results['metadatas'][0][i]['document_id']
                        
                        # Include all public documents
                        if any(document.id == document_id for document in documents):
                            relevant_chunks.append({
                                'document': results['documents'][0][i],
                                'metadata': results['metadatas'][0][i],
                                'distance': results['distances'][0][i]
                            })
                    except (IndexError, KeyError) as e:
                        logger.warning(f"Error processing result at index {i}: {e}")
                        continue
            
            return relevant_chunks
            
        except Exception as e:
            logger.error(f"Error retrieving context: {e}")
            raise AIAssistantError(f"Failed to retrieve context: {e}")
    
    def _generate_response(self, query: str, context_text: str) -> str:
        """
        Generate AI response using OpenAI API.
        """ 
        try:
            system_prompt = f"""
            You are an AI football/soccer expert assistant that provides information ONLY from the uploaded documents.
            Your are currently talking to {self.user_full_name if self.user_full_name else 'User'}.
            Your knowledge comes from a comprehensive collection of documents covering:

            1. Nutrition - Diet and nutritional guidance for footballers
            2. Strength and Conditioning - Physical training and development
            3. Training Program Scheduling - Program design and periodization
            4. General Player Advice - Overall development and career guidance
            5. Injury Prevention and Management - Injury care and prevention strategies
            6. Mental Well-Being - Psychological aspects and mental health
            7. Performance Analytics Tips - Data-driven performance insights
            8. Tactical Development - Game strategy and tactical analysis

            STRICT REQUIREMENTS:
            1. ONLY use information from the provided document excerpts
            2. NEVER make assumptions or use external football knowledge
            3. If the documents don't contain relevant information, clearly state this
            4. Provide specific document citations for every piece of information
            5. Format citations as: [Document Title | Category | Page X]
            6. Focus on practical, actionable advice when possible
            7. Maintain scientific accuracy while being clear and concise
            8. Consider the holistic development of players (physical, mental, tactical)

            If you cannot find relevant information in the documents, respond with:
            "I cannot provide specific information about this topic from the available football documents. Please check other reliable sources or consult with qualified professionals."
            """

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Context:\n{context_text}\n\nQuestion: {query}"}
            ]

            response = self.openai_client.chat.completions.create(
                model="gpt-4o" or "gpt-3.5-turbo",
                messages=messages,
                temperature=0.7,
                max_tokens=1000
            )

            return response.choices[0].message.content

        except Exception as e:
            logger.error(f"Error generating response: {e}")
            raise AIAssistantError(f"Failed to generate response: {e}")
    
    def process_message(self, session_id: str, query_text: str, email: str) -> str:
        try:
            # Set users full name
            self.user_full_name = get_user_full_name(email)
            # Get conversation history
            previous_messages = self.db.query(AI_ChatMessage).filter(
                AI_ChatMessage.session_id == session_id
            ).order_by(AI_ChatMessage.timestamp.desc()).limit(5).all()
            
            # Retrieve relevant context from all books
            relevant_chunks = self._retrieve_relevant_context(email, query_text, n_results=15)
            
            # If no relevant chunks found, return a message indicating no documents
            if not relevant_chunks:
                return "I cannot provide an answer as there are no documents in the knowledge base yet. Please upload some documents first."
            
            # Build context from chunks
            context_text, references = self._build_context_from_chunks(relevant_chunks)
            
            system_prompt = f"""
            You are a highly specialized Football Intelligence Assistant. You are trained solely on a knowledge base made from uploaded scientific PDFs and expert-authored football resources. You do not use any external data or assumptions.
            Your are currently talking to {self.user_full_name if self.user_full_name else 'User'}.
            Your role is to respond accurately, clearly, and professionally to user questions across these domains:

            1. Nutrition
            2. Strength & Conditioning
            3. Training Program Scheduling
            4. General Player Advice
            5. Injury Prevention & Management
            6. Mental Well-Being
            7. Performance Analytics
            8. Tactical Development
            9. General Summaries

            Follow these strict rules:

            1. DO NOT use external football knowledge — ONLY the provided document excerpts.
            2. NEVER guess or assume — if unsure, say:
            "I cannot provide specific information about this topic from the available football documents. Please check other reliable sources or consult with qualified professionals."
            3. Provide citations for all key facts in this format: [Document Title | Category | Page X]
            4. If multiple domains apply, structure the response per domain.
            5. Prioritize actionable advice (e.g., routines, checklists, examples).
            6. Maintain a balance of clarity and scientific accuracy.

            Respond professionally. Always keep the user’s role (coach, player, analyst) in mind if mentioned.
            """



            # Prepare messages for OpenAI
            messages = [{"role": "system", "content": system_prompt}]
            
            # Create conversation history using a proper loop
            conversation_history = []
            for msg in reversed(previous_messages):
                conversation_history.extend([
                    {"role": "user", "content": msg.query_text},
                    {"role": "assistant", "content": msg.response_text}
                ])
            
            # Add conversation history to messages (last 6 messages)
            messages.extend(conversation_history[:6])
            
            # Add current query with context
            user_message = f"""User Query: {query_text}

            Below are the retrieved document excerpts you MUST rely on for answering:

            {context_text}

            Instructions:
            - Use ONLY these sources.
            - DO NOT assume anything beyond them.
            - Cite precisely as: [Document Title | Category | Page X]
            - If information is insufficient, respond with the fallback message.
            """


            messages.append({"role": "user", "content": user_message})
            
            # Generate response using OpenAI
            response = self.openai_client.chat.completions.create(
                model="gpt-4-turbo-preview",  # Using GPT-4 for better responses
                messages=messages,
                max_tokens=1500,
                temperature=0.7
            )
            
            ai_response = response.choices[0].message.content
            
            return ai_response
            
        except Exception as e:
            logger.error(f"Error processing message: {e}")
            raise AIAssistantError(f"Failed to process message: {e}")
    
    def update_session_title(self, session_id: str, query_text: str) -> None:
        """
        Update the chat session title based on the first message.
        
        Args:
            session_id: ID of the session to update
            query_text: First message text to base title on
            
        Raises:
            AIAssistantError: If update fails
        """
        try:
            # Generate a concise title from the query
            title = (
                query_text[:47] + "..." 
                if len(query_text) > 50 
                else query_text
            )
            
            # Update session title
            session = self.db.query(AI_ChatSession).filter(
                AI_ChatSession.id == session_id
            ).first()
            
            if session:
                session.title = title
                session.last_updated = datetime.utcnow()
                self.db.commit()
                logger.info(f"Updated title for session {session_id}: {title}")
            else:
                raise AIAssistantError(f"Session not found: {session_id}")
                
        except Exception as e:
            self.db.rollback()
            logger.error(f"Failed to update session title: {str(e)}")
            raise AIAssistantError(f"Failed to update session title: {str(e)}")

    def _cleanup_orphaned_vectors(self, email: str) -> None:
        """Clean up any orphaned vectors that don't have corresponding database records."""
        try:
            collection = self.get_or_create_collection(email)
            
            # Get all valid embedding IDs from database for public documents
            valid_chunks = (
                self.db.query(AI_DocumentChunk)
                .join(AI_Document)
                .filter(AI_Document.is_public == True)
                .all()
            )
            valid_embedding_ids = {chunk.embedding_id for chunk in valid_chunks if chunk.embedding_id}
            
            # Get all vectors from ChromaDB
            try:
                all_vectors = collection.get()
                if all_vectors and all_vectors['ids']:
                    vector_ids = set(all_vectors['ids'])
                    
                    # Find orphaned vectors (exist in ChromaDB but not in database)
                    orphaned_ids = vector_ids - valid_embedding_ids
                    
                    if orphaned_ids:
                        # Delete orphaned vectors in batches
                        batch_size = 100
                        orphaned_list = list(orphaned_ids)
                        
                        for i in range(0, len(orphaned_list), batch_size):
                            batch = orphaned_list[i:i + batch_size]
                            collection.delete(ids=batch)
                        
                        logger.info(f"Cleaned up {len(orphaned_ids)} orphaned vectors")
                    
            except Exception as e:
                logger.error(f"Error during vector cleanup: {e}")
                
        except Exception as e:
            logger.error(f"Error during orphaned vector cleanup: {e}")
            # Don't raise - this is a cleanup operation

    def reset_user_collection(self, email: str) -> Dict[str, Any]:
        """Reset/clear all vectors for the public collection."""
        try:
            # Delete the public collection
            try:
                self.chroma_client.delete_collection(name="public_documents")
                logger.info("Deleted public collection")
            except Exception as e:
                logger.warning(f"Public collection might not exist: {e}")
            
            # Recreate empty collection
            collection = self.chroma_client.create_collection(
                name="public_documents",
                metadata={"type": "public"}
            )
            
            return {
                "status": "success",
                "message": "Reset public collection",
                "collection_name": "public_documents"
            }
            
        except Exception as e:
            logger.error(f"Error resetting collection: {e}")
            raise AIAssistantError(f"Failed to reset collection: {e}")

    def cleanup_all_orphaned_vectors(self, email: str) -> Dict[str, Any]:
        """Clean up all orphaned vectors in the public collection."""
        try:
            collection = self.get_or_create_collection(email)
            
            # Get all valid embedding IDs from database for public documents
            valid_chunks = (
                self.db.query(AI_DocumentChunk)
                .join(AI_Document)
                .filter(AI_Document.is_public == True)
                .all()
            )
            valid_embedding_ids = {chunk.embedding_id for chunk in valid_chunks if chunk.embedding_id}
            
            # Get all vectors from ChromaDB
            all_vectors = collection.get()
            vector_ids = set(all_vectors['ids']) if all_vectors and all_vectors['ids'] else set()
            
            # Find orphaned vectors
            orphaned_ids = vector_ids - valid_embedding_ids
            
            if orphaned_ids:
                # Delete orphaned vectors in batches
                batch_size = 100
                orphaned_list = list(orphaned_ids)
                deleted_count = 0
                
                for i in range(0, len(orphaned_list), batch_size):
                    batch = orphaned_list[i:i + batch_size]
                    try:
                        collection.delete(ids=batch)
                        deleted_count += len(batch)
                    except Exception as e:
                        logger.error(f"Error deleting batch: {e}")
                
                logger.info(f"Cleaned up {deleted_count} orphaned vectors")
            
            return {
                "total_vectors": len(vector_ids),
                "valid_vectors": len(valid_embedding_ids),
                "orphaned_vectors": len(orphaned_ids),
                "deleted_count": len(orphaned_ids) if orphaned_ids else 0
            }
            
        except Exception as e:
            logger.error(f"Error cleaning orphaned vectors: {e}")
            raise AIAssistantError(f"Failed to cleanup vectors: {e}")

    def get_collection_stats(self, email: str) -> Dict[str, Any]:
        """Get statistics about a user's collection."""
        try:
            collection = self.get_or_create_collection(email)
            
            # Get collection info
            all_vectors = collection.get()
            vector_count = len(all_vectors['ids']) if all_vectors and all_vectors['ids'] else 0
            
            # Get database info
            documents = self.db.query(AI_Document).filter(
                (AI_Document.admin_email == email) | (AI_Document.is_public == True)
            ).all()
            
            chunks = self.db.query(AI_DocumentChunk).filter(
                AI_DocumentChunk.document_id.in_([document.id for document in documents])
            ).all()
            
            return {
                "collection_name": f"user_{email.replace('@', '_').replace('.', '_')}",
                "vector_count": vector_count,
                "documents_count": len(documents),
                "chunks_count": len(chunks),
                "documents": [{"id": document.id, "name": document.document_name} for document in documents]
            }
            
        except Exception as e:
            logger.error(f"Error getting collection stats: {e}")
            raise AIAssistantError(f"Failed to get stats: {e}")

    def cleanup_chroma_db_standalone():
        """Standalone script to clean ChromaDB - run this separately."""
        import chromadb
        from chromadb.config import Settings
        import os
        import shutil
        
        # Option 1: Delete entire ChromaDB directory
        chroma_dir = "./chroma_db"
        if os.path.exists(chroma_dir):
            try:
                shutil.rmtree(chroma_dir)
                print(f"Deleted entire ChromaDB directory: {chroma_dir}")
                
                # Recreate empty directory
                os.makedirs(chroma_dir, exist_ok=True)
                print("Created new empty ChromaDB directory")
            except Exception as e:
                print(f"Error deleting ChromaDB directory: {e}")
        
        # Option 2: Delete specific collections
        try:
            client = chromadb.PersistentClient(
                path="./chroma_db",
                settings=Settings(anonymized_telemetry=False)
            )
            
            # List all collections
            collections = client.list_collections()
            print(f"Found {len(collections)} collections")
            
            for collection in collections:
                try:
                    client.delete_collection(name=collection.name)
                    print(f"Deleted collection: {collection.name}")
                except Exception as e:
                    print(f"Error deleting collection {collection.name}: {e}")
                    
        except Exception as e:
            print(f"Error connecting to ChromaDB: {e}")

    def sync_and_cleanup_database():
        """Sync database and vector store, removing inconsistencies."""
        from app.database import get_db
        from app.models import AI_Document, AI_DocumentChunk
        
        db = next(get_db())
        
        try:
            # Get all users who have documents
            users = db.query(AI_Document.admin_email).distinct().all()
            
            for (email,) in users:
                print(f"Cleaning up for user: {email}")
                
                ai_assistant = AIAssistant(db)
                
                # Reset user collection
                result = ai_assistant.reset_user_collection(email)
                print(f"Reset result: {result}")
                
                # Reprocess all documents for this user
                documents = db.query(AI_Document).filter(AI_Document.admin_email == email).all()
                
                for document in documents:
                    if document.processed:
                        print(f"Reprocessing document: {document.document_name}")
                        
                        # Extract text again
                        try:
                            text_content, _ = ai_assistant._extract_pdf_text(document.file_path)
                            
                            # Reprocess and store chunks
                            ai_assistant._process_and_store_chunks(
                                document.id, email, text_content, document.document_name
                            )
                            
                            print(f"Successfully reprocessed: {document.document_name}")
                        except Exception as e:
                            print(f"Error reprocessing {document.document_name}: {e}")
            
            print("Database sync completed")
            
        except Exception as e:
            print(f"Error during sync: {e}")
        finally:
            db.close()

    def quick_reset_chroma():
        """Quick reset - deletes ChromaDB and recreates empty structure."""
        import os
        import shutil
        
        chroma_path = "./chroma_db"
        
        print("Stopping application first...")
        
        # Delete ChromaDB directory
        if os.path.exists(chroma_path):
            try:
                shutil.rmtree(chroma_path)
                print(f"✅ Deleted ChromaDB directory: {chroma_path}")
            except Exception as e:
                print(f"❌ Error deleting directory: {e}")
                return False
        
        # Recreate directory
        try:
            os.makedirs(chroma_path, exist_ok=True)
            print(f"✅ Created new ChromaDB directory: {chroma_path}")
            
            # Test initialization
            import chromadb
            from chromadb.config import Settings
            
            client = chromadb.PersistentClient(
                path=chroma_path,
                settings=Settings(anonymized_telemetry=False)
            )
            
            print("✅ ChromaDB initialized successfully")
            print("🔄 Restart your application to begin fresh")
            
            return True
            
        except Exception as e:
            print(f"❌ Error creating new ChromaDB: {e}")
            return False
    
    def verify_knowledge_base_consistency(self, email: str) -> dict:
        """
        Verify the consistency between application database and vector store
        
        Args:
            email: Admin email to check documents for
            
        Returns:
            Dict containing consistency check results
        """
        try:
            # Get all documents from application database
            db_documents = (self.db.query(AI_Document)
                    .filter((AI_Document.admin_email == email) | (AI_Document.is_public == True))
                    .all())
            db_document_ids = set(str(document.id) for document in db_documents)
            
            # Get all chunks from database
            db_chunks = (self.db.query(AI_DocumentChunk)
                        .join(AI_Document)
                        .filter((AI_Document.admin_email == email) | (AI_Document.is_public == True))
                        .all())
            chunk_document_ids = set(str(chunk.document_id) for chunk in db_chunks)
            
            # Get embeddings from vector store using get_or_create_collection
            collection = self.get_or_create_collection(email)
            vector_ids = set()
            if collection is not None:
                all_vectors = collection.get()
                if all_vectors and all_vectors['ids']:
                    vector_ids = set(str(id) for id in all_vectors['ids'])
            
            # Check consistency
            is_consistent = (db_document_ids == chunk_document_ids) and all(
                chunk.embedding_id in vector_ids for chunk in db_chunks if chunk.embedding_id
            )
            
            return {
                "is_consistent": is_consistent,
                "database_documents": len(db_document_ids),
                "chunk_documents": len(chunk_document_ids),
                "vector_count": len(vector_ids)
            }
            
        except Exception as e:
            logger.error(f"Error checking consistency: {e}")
            raise AIAssistantError(f"Failed to verify consistency: {e}")
    def delete_documents(self, email: str, document_ids: List[str]) -> int:
        """
        Delete documents and their associated data
        
        Args:
            email: Admin email requesting deletion
            document_ids: List of document IDs to delete
            
        Returns:
            Number of documents deleted
        """
        deleted_count = 0
        
        try:
            # Get collection once for all operations
            collection = self.get_or_create_collection(email)
            
            for document_id in document_ids:
                # Verify document exists and user has permission
                document = (self.db.query(AI_Document)
                    .filter(AI_Document.id == document_id, AI_Document.admin_email == email)
                    .first())
                
                if not document:
                    continue
                    
                # Delete physical file
                if os.path.exists(document.file_path):
                    os.remove(document.file_path)
                    logger.info(f"Deleted physical file: {document.file_path}")
                
                # Get chunks to delete from vector store
                chunks = self.db.query(AI_DocumentChunk).filter(AI_DocumentChunk.document_id == document_id).all()
                chunk_ids = [chunk.id for chunk in chunks]
                embedding_ids = [chunk.embedding_id for chunk in chunks if chunk.embedding_id]
                
                # Delete from vector store
                if collection is not None and embedding_ids:
                    collection.delete(ids=embedding_ids)
                
                # Delete chunks from database
                if chunk_ids:
                    self.db.query(AI_DocumentChunk).filter(AI_DocumentChunk.id.in_(chunk_ids)).delete()
                
                # Delete document record
                self.db.query(AI_Document).filter(AI_Document.id == document_id).delete()
                deleted_count += 1
                
                logger.info(f"Deleted document {document.document_name} (ID: {document_id})")
                
        except Exception as e:
            logger.error(f"Error deleting documents: {e}")
            raise AIAssistantError(f"Failed to delete documents: {e}")
            
        return deleted_count