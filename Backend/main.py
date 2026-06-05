from fastapi import FastAPI, UploadFile, File, Query
from fastapi.middleware.cors import CORSMiddleware
import shutil
import os
from dotenv import load_dotenv

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = "uploads"
CHROMA_DIR = "Chroma_DB"

os.makedirs(UPLOAD_DIR, exist_ok=True)

embedding_model = FastEmbedEmbeddings(
    model_name="BAAI/bge-small-en-v1.5"
)


@app.get("/")
def home():
    return {
        "message": "Backend Running Successfully"
    }


@app.get("/health")
def health():
    return {
        "status": "ok"
    }


@app.post("/uploads")
async def upload_pdf(file: UploadFile = File(...)):

    try:
        if not file.filename.endswith(".pdf"):
            return {
                "error": "Please upload only PDF file."
            }

        # Remove old Chroma DB before uploading new PDF
        if os.path.exists(CHROMA_DIR):
            shutil.rmtree(CHROMA_DIR)

        safe_filename = file.filename.replace(" ", "_")
        file_path = os.path.join(UPLOAD_DIR, safe_filename)

        with open(file_path, "wb") as f:
            shutil.copyfileobj(file.file, f)

        loader = PyPDFLoader(file_path)
        docs = loader.load()

        if not docs:
            return {
                "error": "No text found in PDF."
            }

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=100
        )

        chunks = splitter.split_documents(docs)

        Chroma.from_documents(
            documents=chunks,
            embedding=embedding_model,
            persist_directory=CHROMA_DIR
        )

        return {
            "msg": "PDF Uploaded Successfully"
        }

    except Exception as e:
        return {
            "error": f"Upload failed: {str(e)}"
        }


@app.post("/ask")
def ask_question(question: str = Query(...)):

    try:
        if not GROQ_API_KEY:
            return {
                "answer": "GROQ_API_KEY is missing in Render environment variables."
            }

        if not os.path.exists(CHROMA_DIR):
            return {
                "answer": "Please upload a PDF first."
            }

        db = Chroma(
            persist_directory=CHROMA_DIR,
            embedding_function=embedding_model
        )

        retriever = db.as_retriever(
            search_kwargs={"k": 3}
        )

        docs = retriever.invoke(question)

        if not docs:
            return {
                "answer": "No relevant information found in PDF."
            }

        context = "\n\n".join(
            [doc.page_content for doc in docs]
        )

        llm = ChatGroq(
            model="llama-3.3-70b-versatile",
            api_key=GROQ_API_KEY
        )

        prompt = f"""
Answer the question using only the given context.

Context:
{context}

Question:
{question}

Answer:
"""

        response = llm.invoke(prompt)

        return {
            "question": question,
            "answer": response.content
        }

    except Exception as e:
        return {
            "answer": f"Backend error: {str(e)}"
        }