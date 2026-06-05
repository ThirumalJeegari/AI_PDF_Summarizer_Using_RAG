from fastapi import FastAPI, UploadFile, File, Query
import shutil
import os
from dotenv import load_dotenv

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq

load_dotenv()

# In Render, add GROQ_API_KEY in Environment Variables
# api_key is also supported for your old local .env name
GROQ_API_KEY = os.getenv("GROQ_API_KEY") or os.getenv("api_key")

app = FastAPI()

os.makedirs("uploads", exist_ok=True)


@app.get("/")
def home():
    return {
        "message": "Backend Running Successfully"
    }


@app.post("/uploads")
async def upload_pdf(file: UploadFile = File(...)):

    file_path = f"uploads/{file.filename}"

    with open(file_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    loader = PyPDFLoader(file_path)
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    chunks = splitter.split_documents(docs)

    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory="Chroma_DB"
    )

    return {
        "msg": "PDF Uploaded Successfully"
    }


@app.post("/ask")
def ask_question(question: str = Query(...)):

    if not GROQ_API_KEY:
        return {
            "answer": "GROQ_API_KEY is missing. Add it in Render Environment Variables."
        }

    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    db = Chroma(
        persist_directory="Chroma_DB",
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
