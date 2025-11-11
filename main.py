# main.py
import uvicorn
import os
from dotenv import load_dotenv
from app.api import app

# Load environment variables for upload configuration
load_dotenv()
load_dotenv('.env.upload')

# Get upload configuration from environment
MAX_UPLOAD_SIZE_MB = int(os.getenv("MAX_UPLOAD_SIZE_MB", "500"))
REQUEST_TIMEOUT = int(os.getenv("REQUEST_TIMEOUT", "300"))
KEEPALIVE_TIMEOUT = int(os.getenv("KEEPALIVE_TIMEOUT", "65"))

if __name__ == "__main__":
    # Configure uvicorn with proper settings for large file uploads
    uvicorn.run(
        "app.api:app",
        host="0.0.0.0",
        port=8001,
        reload=True,
        # Increase timeouts for large file processing
        timeout_keep_alive=KEEPALIVE_TIMEOUT,  # Keep connection alive longer
        timeout_graceful_shutdown=40,  # Graceful shutdown timeout
        # Handle larger payloads - CRITICAL for large uploads
        limit_max_requests=None,
        limit_concurrency=None,
    ) 