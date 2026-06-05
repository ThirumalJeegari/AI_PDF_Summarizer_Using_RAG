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

os.makedirs("uploads", exist_ok=True)

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

        Chroma.from_documents(
            documents=chunks,
            embedding=embedding_model,
            persist_directory="Chroma_DB"
        )

        return {
            "msg": "PDF Uploaded Successfully"
        }

    except Exception as e:
        return {
            "error": str(e)
        }


@app.post("/ask")
def ask_question(question: str = Query(...)):

    try:
        if not GROQ_API_KEY:
            return {
                "answer": "GROQ_API_KEY is missing in Render environment variables."
            }

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

    except Exception as e:
        return {
            "answer": f"Backend error: {str(e)}"
        }