# Football AI Assistant

A comprehensive AI-powered assistant API designed specifically for football (soccer) analysis, player development, and tactical insights. The system combines modern AI technologies with football expertise to provide intelligent responses to queries related to nutrition, training, tactical development, injury prevention, and performance analytics.

## 🚀 Features

- **AI-Powered Chat System**: Interactive conversations with specialized football knowledge
- **Document Management**: Upload and process PDF documents to enhance the AI's knowledge base
- **Session Management**: Organized chat sessions with historical tracking
- **User Authentication**: Secure user management with staff-level document access
- **Knowledge Base**: Specialized categories including:
  - Nutrition
  - Strength and Conditioning
  - Training Program Schedules
  - General Player Advice
  - Injury Prevention and Management
  - Mental Well-Being
  - Performance Analytics Tips
  - Tactical Development

## 🛠 Technology Stack

- **Backend**: FastAPI (Python)
- **Database**: PostgreSQL with SQLAlchemy ORM
- **AI/ML**: OpenAI GPT integration
- **Vector Database**: ChromaDB for document embeddings
- **Authentication**: JWT-based user authentication
- **File Processing**: PyPDF2 for document parsing
- **Database Migrations**: Alembic

## 📋 Prerequisites

- Python 3.8+
- PostgreSQL database
- OpenAI API key
- Node.js (for frontend, if applicable)

## 🔧 Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/your-username/football_ai.git
   cd football_ai
   ```

2. **Install dependencies**
   ```bash
   pip install -r ai_requirements.txt
   ```

3. **Environment Setup**
   Create a `.env` file in the root directory:
   ```env
   OPENAI_API_KEY=your_openai_api_key_here
   DATABASE_URL=postgresql://username:password@localhost:5432/football_ai
   ALLOWED_ORIGINS=http://localhost:3000,http://localhost:5173
   ```

4. **Database Setup**
   ```bash
   # Run database migrations
   alembic upgrade head
   ```

5. **Start the application**
   ```bash
   python main.py
   ```

   The API will be available at `http://127.0.0.1:8001`

## 📚 API Documentation

Once the server is running, access the interactive API documentation at:
- Swagger UI: `http://127.0.0.1:8001/docs`
- ReDoc: `http://127.0.0.1:8001/redoc`

## 🔌 Main API Endpoints

### Chat System
- `POST /ai/chat_session/` - Create a new chat session
- `POST /ai/chat/` - Send a message and get AI response
- `GET /ai/all_chat/` - Retrieve all messages from a session
- `GET /ai/all_sessions/` - Get all user sessions grouped by date
- `GET /ai/search/` - Search sessions by title

### Document Management (Staff Only)
- `POST /ai/upload/` - Upload and process PDF documents
- `GET /ai/document_list/` - List all documents
- `DELETE /ai/delete_documents/` - Delete documents from knowledge base

### Session Management
- `DELETE /ai/delete_session_chats/` - Delete all messages from a session

## 🏗 Project Structure

```
football_ai/
├── app/
│   ├── ai_assistant.py      # Core AI logic and document processing
│   ├── api.py              # FastAPI routes and endpoints
│   ├── database.py         # Database configuration and utilities
│   └── models.py           # SQLAlchemy database models
├── migrations/             # Alembic database migrations
├── uploads/               # Document storage directory
├── chroma_db/            # ChromaDB vector database
├── assets/               # Project diagrams and documentation
├── main.py              # Application entry point
├── ai_requirements.txt  # Python dependencies
└── alembic.ini         # Alembic configuration
```

## 🎯 Key Features

### AI Assistant Capabilities
- **Contextual Understanding**: Maintains conversation context across chat sessions
- **Document-Enhanced Responses**: Uses uploaded documents to provide more accurate answers
- **Category-Specific Knowledge**: Specialized responses for different football aspects
- **User Personalization**: Adapts responses based on user information and history

### Security Features
- User authentication and authorization
- Staff-only document management
- Session-based access control
- Input validation and sanitization

### Performance Optimizations
- Caching for frequent queries
- Efficient vector similarity search
- Optimized database queries
- Background document processing

## 🔐 Authentication

The system uses a two-tier authentication approach:
1. **User Authentication**: Basic user verification for chat access
2. **Staff Authorization**: Enhanced permissions for document management

## 📖 Usage Examples

### Creating a Chat Session
```python
import requests

response = requests.post(
    "http://127.0.0.1:8001/ai/chat_session/",
    json={"email": "user@example.com"}
)
session_id = response.json()["session_id"]
```

### Sending a Chat Message
```python
response = requests.post(
    "http://127.0.0.1:8001/ai/chat/",
    json={
        "session_id": session_id,
        "query_text": "What are the best nutrition practices for football players?",
        "email": "user@example.com"
    }
)
```

## 🚀 Development

### Running in Development Mode
```bash
python main.py
```

### Database Migrations
```bash
# Create a new migration
alembic revision --autogenerate -m "Description of changes"

# Apply migrations
alembic upgrade head
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🆘 Support

For support and questions, please open an issue in the GitHub repository or contact the development team.

## 🔄 Version History

- **v1.0.0** - Initial release with core AI chat functionality
- **v1.1.0** - Added document management and enhanced knowledge base
- **v1.2.0** - Improved user authentication and session management

---

**Built with ❤️ for the football community**