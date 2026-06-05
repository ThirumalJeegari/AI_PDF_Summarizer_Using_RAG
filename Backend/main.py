from fastapi import FastAPI, UploadFile, File, Query
from fastapi.middleware.cors import CORSMiddleware
import os
import shutil
from dotenv import load_dotenv
from pypdf import PdfReader
import chromadb
from fastembed import TextEmbedding
from groq import Groq

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
COLLECTION_NAME = "pdf_collection"

os.makedirs(UPLOAD_DIR, exist_ok=True)

embedding_model = TextEmbedding(
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


def split_text(text, chunk_size=800, chunk_overlap=100):
    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]

        if chunk.strip():
            chunks.append(chunk)

        start = end - chunk_overlap

    return chunks


@app.post("/uploads")
async def upload_pdf(file: UploadFile = File(...)):

    try:
        if not file.filename.lower().endswith(".pdf"):
            return {
                "error": "Please upload only PDF files."
            }

        if os.path.exists(CHROMA_DIR):
            shutil.rmtree(CHROMA_DIR)

        safe_filename = file.filename.replace(" ", "_")
        file_path = os.path.join(UPLOAD_DIR, safe_filename)

        with open(file_path, "wb") as f:
            shutil.copyfileobj(file.file, f)

        reader = PdfReader(file_path)

        full_text = ""

        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                full_text += page_text + "\n"

        if not full_text.strip():
            return {
                "error": "No text found in PDF."
            }

        chunks = split_text(full_text)

        client = chromadb.PersistentClient(path=CHROMA_DIR)

        collection = client.get_or_create_collection(
            name=COLLECTION_NAME
        )

        embeddings = list(
            embedding_model.embed(chunks)
        )

        ids = [f"chunk_{i}" for i in range(len(chunks))]

        collection.add(
            documents=chunks,
            embeddings=[embedding.tolist() for embedding in embeddings],
            ids=ids
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

        client = chromadb.PersistentClient(path=CHROMA_DIR)

        collection = client.get_collection(
            name=COLLECTION_NAME
        )

        question_embedding = list(
            embedding_model.embed([question])
        )[0]

        results = collection.query(
            query_embeddings=[question_embedding.tolist()],
            n_results=3
        )

        documents = results["documents"][0]

        if not documents:
            return {
                "answer": "No relevant information found in PDF."
            }

        context = "\n\n".join(documents)

        groq_client = Groq(
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

        response = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )

        answer = response.choices[0].message.content

        return {
            "question": question,
            "answer": answer
        }

    except Exception as e:
        return {
            "answer": f"Backend error: {str(e)}"
        }