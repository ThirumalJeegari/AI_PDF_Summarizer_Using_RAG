from fastapi import FastAPI, UploadFile, File
from fastapi import Query
import shutil
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_groq import ChatGroq

app = FastAPI()

@app.get("/")
def home():
    return {
        "message": "Backend Running Successfully"
    }


@app.post("/uploads")
async def upload_pdf(file: UploadFile = File(...)):

    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_community.vectorstores import Chroma

    with open(file.filename, "wb") as f:
        shutil.copyfileobj(file.file, f)
    loader = PyPDFLoader(file.filename)

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


    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_community.vectorstores import Chroma

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

    docs = retriever.get_relevant_documents(question)

    context = "\n\n".join([doc.page_content for doc in docs])

    llm = ChatGroq(
        model="llama-3.3-70b-versatile"
    )

    prompt = f"""
    Answer the question using only the provided context.

    Context:
    {context}

    Question:
    {question}

    Answer:
    """

    answer = llm.invoke(prompt)

    return {
        "question": question,
        "answer": answer
    }