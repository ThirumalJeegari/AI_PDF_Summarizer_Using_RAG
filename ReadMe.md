# AI PDF Summarizer Using RAG

This project is an AI-powered PDF chatbot built using **FastAPI**, **Streamlit**, **LangChain**, **ChromaDB**, and **Groq LLM**.
Users can upload a PDF file and ask questions based on the PDF content. The backend extracts text from the PDF, creates embeddings, stores them in a vector database, retrieves relevant content, and generates answers using Groq AI.

## Project Structure

```text
AI_PDF_Summarizer_Using_RAG/
│
├── Backend/
│   ├── main.py
│   ├── requirements.txt
│   ├── runtime.txt
│
├── Frontend/
│   ├── app.py
│   ├── requirements.txt
│
├── .gitignore
└── README.md
```

## Features

* Upload PDF documents
* Extract and process PDF text
* Split text into chunks
* Store embeddings in ChromaDB
* Ask questions from uploaded PDF
* Generate AI answers using Groq LLM
* Separate backend and frontend deployment
* Render deployment supported

## Tech Stack

### Backend

* FastAPI
* LangChain
* ChromaDB
* Groq API
* PyPDF
* FastEmbed / Embeddings
* Uvicorn

### Frontend

* Streamlit
* Requests

## Backend Setup

Go to the backend folder:

```bash
cd Backend
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run backend locally:

```bash
uvicorn main:app --reload
```

Backend will run at:

```text
http://localhost:8000
```

Test backend:

```text
http://localhost:8000/
```

Expected response:

```json
{
  "message": "Backend Running Successfully"
}
```

## Frontend Setup

Open a new terminal and go to the frontend folder:

```bash
cd Frontend
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run frontend locally:

```bash
streamlit run app.py
```

Frontend will run at:

```text
http://localhost:8501
```

## Environment Variables

Create a `.env` file locally if needed.

Example:

```env
GROQ_API_KEY=your_groq_api_key
BACKEND_URL=http://localhost:8000
```

Do not upload `.env` to GitHub.

## Important Security Note

Never push API keys to GitHub.

Add these lines in `.gitignore`:

```text
.env
__pycache__/
*.pyc
Chroma_DB/
uploads/
```

If an API key is accidentally pushed or detected by GitHub, delete that key and create a new one.

## Backend Requirements

Example `Backend/requirements.txt`:

```txt
fastapi==0.115.6
uvicorn[standard]==0.32.1
python-dotenv==1.0.1
python-multipart==0.0.20

langchain==0.3.13
langchain-community==0.3.13
langchain-text-splitters==0.3.4
langchain-groq==0.2.2

chromadb==0.5.23
fastembed==0.4.2
pypdf==5.1.0
```

## Frontend Requirements

Example `Frontend/requirements.txt`:

```txt
streamlit==1.40.2
requests==2.32.3
```

## Runtime File

Create this file inside the Backend folder:

```text
Backend/runtime.txt
```

Add only this line:

```text
python-3.11.9
```

Do not add API keys inside `runtime.txt`.

## Render Deployment

This project needs two Render services:

1. Backend service for FastAPI
2. Frontend service for Streamlit

Deploy backend first, then frontend.

## Deploy Backend on Render

Create a new Web Service on Render.

Use these settings:

```text
Root Directory:
Backend
```

```text
Build Command:
pip install -r requirements.txt
```

```text
Start Command:
uvicorn main:app --host 0.0.0.0 --port $PORT
```

Add environment variables:

```env
PYTHON_VERSION=3.11.9
GROQ_API_KEY=your_groq_api_key
```

After deployment, Render will give a backend URL like:

```text
https://your-backend-service-name.onrender.com
```

Test it in the browser:

```text
https://your-backend-service-name.onrender.com/
```

## Deploy Frontend on Render

Create another Web Service on Render.

Use these settings:

```text
Root Directory:
Frontend
```

```text
Build Command:
pip install -r requirements.txt
```

```text
Start Command:
streamlit run app.py --server.port $PORT --server.address 0.0.0.0
```

Add environment variable:

```env
BACKEND_URL=https://your-backend-service-name.onrender.com
```

Use your real backend Render URL.

## Application Flow

```text
User uploads PDF
        ↓
Streamlit frontend sends PDF to FastAPI backend
        ↓
Backend extracts PDF text
        ↓
Text is split into chunks
        ↓
Embeddings are created
        ↓
Chunks are stored in ChromaDB
        ↓
User asks a question
        ↓
Backend retrieves relevant PDF content
        ↓
Groq LLM generates answer
        ↓
Answer is shown in Streamlit frontend
```

## Common Errors and Fixes

### Backend connection error: localhost refused

This happens when frontend is using:

```python
http://localhost:8000
```

On Render, use:

```env
BACKEND_URL=https://your-backend-service-name.onrender.com
```

### GitHub push blocked because of API key

Remove the API key from your files and commit history.
Never store secrets in GitHub.

### Tokenizers build error

Use Python 3.11.9 by adding:

```env
PYTHON_VERSION=3.11.9
```

Also add:

```text
Backend/runtime.txt
```

with:

```text
python-3.11.9
```

### Response ended prematurely

This can happen when the backend crashes due to memory.
Use lighter embeddings such as FastEmbed instead of heavy sentence-transformers.


Frontend_URL(Render)= https://ai-pdf-summarizer-using-rag-frontend.onrender.com/

Backend URL(Render) = https://ai-pdf-summarizer-using-rag.onrender.com/


## Author

**Jeegari Thirumal**

