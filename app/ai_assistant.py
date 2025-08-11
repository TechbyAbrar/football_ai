import os
import logging
import uuid
from typing import List, Dict, Any, Optional, Tuple, cast
from datetime import datetime
import tempfile
import shutil
from functools import lru_cache
import hashlib

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
        
        # Initialize caching for performance optimization
        self._document_cache = {}
        self._embedding_cache = {}
        self._cache_timestamp = datetime.now()

        
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

    def _get_cache_key(self, prefix: str) -> str:
        """Generate cache key with hourly refresh."""
        return f"{prefix}_{datetime.now().hour}"
    
    def _get_cached_embedding(self, text: str) -> List[float]:
        """Cache embeddings to avoid recomputation."""
        text_hash = hashlib.md5(text.encode()).hexdigest()
        if text_hash not in self._embedding_cache:
            self._embedding_cache[text_hash] = self.embedding_model.encode(text).tolist()
        return self._embedding_cache[text_hash]
    
    def _get_cached_documents(self) -> List[AI_Document]:
        """Cache public documents with hourly refresh."""
        cache_key = self._get_cache_key("public_docs")
        if cache_key not in self._document_cache:
            self._document_cache[cache_key] = self.db.query(AI_Document).filter(
                AI_Document.is_public == True
            ).all()
        return self._document_cache[cache_key]

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
            
            # Process chunks in batches for better performance - increased batch size
            batch_size = 100  # Increased from 50 for better performance
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

    def _retrieve_relevant_context(self, email: str, query: str, n_results: int = 5) -> List[Dict]:
        """
        Retrieve relevant context from all public documents with caching optimization.
        Reduced default n_results from 10 to 5 for better performance.
        """
        try:
            # Get public collection
            collection = self.get_or_create_collection(email)
            
            # Get cached public documents instead of querying every time
            documents = self._get_cached_documents()
            
            # If no documents exist, return empty list
            if not documents:
                return []
            
            # Get cached query embedding
            query_embedding = self._get_cached_embedding(query)
            
            # Search for similar chunks with metadata filtering for better performance
            results = collection.query(
                query_embeddings=[query_embedding],
                n_results=min(n_results, len(documents)),  # Don't request more results than documents
                where={"is_public": "true"},  # Pre-filter in ChromaDB for performance
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
    

    def process_message(self, session_id: str, query_text: str, email: str) -> str:
        try:
            # Set users full name
            self.user_full_name = get_user_full_name(email)
            # Get conversation history - reduced from 5 to 3 for better performance
            previous_messages = self.db.query(AI_ChatMessage).filter(
                AI_ChatMessage.session_id == session_id
            ).order_by(AI_ChatMessage.timestamp.desc()).limit(3).all()
            
            # Retrieve relevant context from all books - reduced from 15 to 8 for better performance
            relevant_chunks = self._retrieve_relevant_context(email, query_text, n_results=8)
            
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
            
            # Create conversation history using a proper loop - reduced to last 4 messages for performance
            conversation_history = []
            for msg in reversed(previous_messages):
                conversation_history.extend([
                    {"role": "user", "content": msg.query_text},
                    {"role": "assistant", "content": msg.response_text}
                ])
            
            # Add conversation history to messages (last 4 messages instead of 6)
            messages.extend(conversation_history[:4])
            
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
                max_tokens=1200,  # Reduced from 1500 for faster responses
                temperature=0.7
            )
            
            ai_response = response.choices[0].message.content
            
            return ai_response
            
        except Exception as e:
            logger.error(f"Error processing message: {e}")
            raise AIAssistantError(f"Failed to process message: {e}")

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