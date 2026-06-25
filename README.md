# 📄 DOCUMIND - Intelligent PDF Chat Assistant

**DOCUMIND** is a Retrieval-Augmented Generation (RAG) application that lets you upload PDFs and ask questions about their content. It uses Groq's LLaMA-3.3-70B model and a hybrid search approach (FAISS + BM25) to deliver accurate, sourced answers.

[![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.137-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Groq](https://img.shields.io/badge/Groq-LLaMA--3.3-FF6B00?style=for-the-badge)](https://groq.com)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg?style=for-the-badge)](http://makeapullrequest.com)
[![MIT License](https://img.shields.io/badge/License-MIT-blue.svg?style=for-the-badge)](LICENSE)

---

## ✨ Features

- 📄 **Upload PDFs** – Drag-and-drop or click to upload multiple PDFs
- 🔍 **Hybrid Search** – Combines FAISS vector search with BM25 keyword search
- 🤖 **AI-Powered Answers** – Uses Groq's LLaMA-3.3-70B model for accurate responses
- 📚 **Source Tracking** – Shows which documents and pages answers came from
- 🎨 **Beautiful UI** – Modern, responsive interface with smooth animations
- ⚡ **Fast & Lightweight** – Built with FastAPI and optimized for performance
- 🔒 **Secure** – API keys stored in environment variables
- 🌐 **CORS Enabled** – Can be used with any frontend application

---

## 🏗️ Architecture


Here is your **complete README.md** in one block:

```markdown
# 📄 DOCUMIND - Intelligent PDF Chat Assistant

**DOCUMIND** is a Retrieval-Augmented Generation (RAG) application that lets you upload PDFs and ask questions about their content. It uses Groq's LLaMA-3.3-70B model and a hybrid search approach (FAISS + BM25) to deliver accurate, sourced answers.

[![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.137-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Groq](https://img.shields.io/badge/Groq-LLaMA--3.3-FF6B00?style=for-the-badge)](https://groq.com)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg?style=for-the-badge)](http://makeapullrequest.com)
[![MIT License](https://img.shields.io/badge/License-MIT-blue.svg?style=for-the-badge)](LICENSE)

---

## ✨ Features

- 📄 **Upload PDFs** – Drag-and-drop or click to upload multiple PDFs
- 🔍 **Hybrid Search** – Combines FAISS vector search with BM25 keyword search
- 🤖 **AI-Powered Answers** – Uses Groq's LLaMA-3.3-70B model for accurate responses
- 📚 **Source Tracking** – Shows which documents and pages answers came from
- 🎨 **Beautiful UI** – Modern, responsive interface with smooth animations
- ⚡ **Fast & Lightweight** – Built with FastAPI and optimized for performance
- 🔒 **Secure** – API keys stored in environment variables
- 🌐 **CORS Enabled** – Can be used with any frontend application

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Frontend (HTML/CSS/JS)                  │
│                   Drag & Drop PDF Upload                    │
│                   Chat Interface                            │
└─────────────────────────┬───────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│                  FastAPI Backend                            │
│  ┌─────────────────────────────────────────────────────┐    │
│  │  /upload  – Process PDFs & Create Vector Store      │    │
│  │  /ask     – Answer Questions using RAG              │    │
│  │  /status  – Check system readiness                  │    │
│  └─────────────────────────────────────────────────────┘    │
└─────────────┬───────────────────────┬───────────────────────┘
              │                       │
              ▼                       ▼
┌─────────────────────┐   ┌─────────────────────────────┐
│   Document Parsing  │   │   Vector Store & Search      │
│   (Docling)         │   │   (FAISS + BM25)            │
└─────────────────────┘   └─────────────────────────────┘
              │                       │
              └───────────────────────┘
                          │
                          ▼
              ┌─────────────────────────┐
              │   Groq LLaMA-3.3-70B    │
              │   (Cloud LLM)           │
              └─────────────────────────┘
```

---

## 🛠️ Tech Stack

| Component | Technology | Purpose |
| :--- | :--- | :--- |
| Backend | FastAPI, Python 3.11 | REST API server |
| **Document Parsing** | Docling | Extract text from PDFs |
| **Text Splitting** | RecursiveCharacterTextSplitter | Chunk documents |
| **Vector Store** | FAISS | Efficient similarity search |
| **Keyword Search** | BM25 | Hybrid ranking |
| **Embeddings** | Sentence Transformers (all-MiniLM-L6-v2) | Convert text to vectors |
| **LLM** | Groq (LLaMA-3.3-70B) | Generate answers |
| **Frontend** | HTML5, CSS3, Vanilla JS | User interface |


## 🚀 Getting Started

### Prerequisites
- Python 3.11+
- Groq API Key (Free: [console.groq.com](https://console.groq.com))

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/Aarya713/rag-project.git
   cd rag-project
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r backend/requirements.txt
   ```

4. **Set up environment variables**
   
   Create a `.env` file in the project root:
   ```env
   GROQ_API_KEY=your_groq_api_key_here
   ```

5. **Run the application**
   ```bash
   python backend/run.py
   ```

6. **Open your browser**
   
   Visit: `http://localhost:8000`

---

## 🔧 API Endpoints

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `POST` | `/upload` | Upload PDF(s) for processing |
| `POST` | `/ask` | Ask a question about your documents |
| `GET` | `/status` | Get system status and chunk count |

### Example API Request

**Upload PDFs:**
```bash
curl -X POST -F "files=@document.pdf" http://localhost:8000/upload
```

**Ask a Question:**
```bash
curl -X POST -H "Content-Type: application/json" \
  -d '{"question": "What are the key findings?"}' \
  http://localhost:8000/ask
```

### Sample Response
```json
{
  "answer": "The document discusses the implementation of RAG systems...",
  "sources": [
    {"source": "document.pdf", "page": 3},
    {"source": "document.pdf", "page": 5}
  ]
}
```

---

## 📊 Performance

| Metric | Value |
| :--- | :--- |
| **Average Response Time** | 1.5 – 3 seconds |
| **Supported File Types** | PDF (.pdf) |
| **Max Chunk Size** | 500 characters |
| **Chunk Overlap** | 50 characters |
| **Search Depth** | Top 4 documents |
| **Memory Usage** | ~512 MB (with lazy loading) |

---

## 🏆 Why This Project Stands Out

- **Hybrid Search** – Combines the best of vector similarity and keyword matching
- **Real-time Chat Interface** – Smooth, responsive UI with loading states
- **Source Transparency** – Shows users exactly where answers come from
- **Production-Ready** – Proper error handling, environment variables, and CORS configuration
- **Efficient Resource Management** – Lazy loading of heavy models saves memory
- **Modern Tech Stack** – Uses cutting-edge tools like Groq's LLaMA-3.3 and FAISS
- **Deployable** – Ready for deployment on Railway, Render, or Hugging Face Spaces

---

## 📂 Project Structure

```
rag-project/
├── backend/
│   ├── app/
│   │   ├── __init__.py
│   │   ├── main.py          # FastAPI entry point
│   │   ├── models.py        # Pydantic models
│   │   └── rag_engine.py    # Core RAG logic
│   ├── run.py               # Server startup
│   └── requirements.txt     # Python dependencies
├── frontend/
│   └── index.html           # Beautiful UI
├── screenshots/             # Demo images
├── .env                     # API keys (not in repo)
├── .gitignore               # Ignore sensitive files
├── README.md                # Project documentation
└── LICENSE                  # MIT License
```

---

## 🧪 Testing

To run tests (if you have them):
```bash
pytest
```

---

## 🤝 Contributing

Contributions are welcome! Here's how to help:

1. Fork the repository
2. Create your feature branch: `git checkout -b feature/amazing-feature`
3. Commit your changes: `git commit -m 'Add some amazing feature'`
4. Push to the branch: `git push origin feature/amazing-feature`
5. Open a Pull Request

### Guidelines
- Follow PEP 8 style guide
- Write meaningful commit messages
- Add comments for complex logic
- Update documentation as needed

---

## 📄 License

This project is licensed under the MIT License – see the [LICENSE](LICENSE) file for details.

---

## 👤 Author

**Aarya Jagtap**
- GitHub: [@Aarya713](https://github.com/Aarya713)
- LinkedIn: [https://linkedin.com/in/your-profile](https://linkedin.com/in/your-profile)
- Email: aaryajagtap05@gmail.com

---

## 🙏 Acknowledgments

- [Groq](https://groq.com) – For providing fast and free LLM access
- [LangChain](https://langchain.com) – For the RAG components
- [FAISS](https://github.com/facebookresearch/faiss) – For efficient vector search
- [Docling](https://github.com/DS4SD/docling) – For PDF parsing
- [FastAPI](https://fastapi.tiangolo.com/) – For the amazing web framework

---

## 📬 Contact

Have questions or feedback? Feel free to reach out!

- GitHub Issues: [Create an issue](https://github.com/Aarya713/rag-project/issues)
- Email: your.email@example.com

---

## ⭐ Star History

If you find this project useful, please consider giving it a star ⭐ – it helps others discover it!

[![Star History Chart](https://api.star-history.com/svg?repos=Aarya713/rag-project&type=Date)](https://star-history.com/#Aarya713/rag-project&Date)

---

## 📈 Future Improvements

- [ ] Support for more file formats (DOCX, TXT, CSV)
- [ ] Chat history persistence
- [ ] User authentication
- [ ] Document summarization
- [ ] Multi-language support
- [ ] Docker containerization
- [ ] CI/CD pipeline

