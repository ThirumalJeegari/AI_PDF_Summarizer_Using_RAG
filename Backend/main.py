from fastapi import FastAPI,UploadFile
import shutil
import langchain_community.document_loaders
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader


app = FastAPI()

@app.get("/")
def home():
    return{
        "message":"Backend Running Successfully.."
    }

@app.post("/uploads")
async def receving_incoming_request(file:UploadFile):
    with open(file.filename,"wb") as f:
        shutil.copyfileobj(file.file,f)

        loader =PyPDFLoader(file.filename)
        docs = loader.load()

        splits = RecursiveCharacterTextSplitter(
            chunk_size = 1000,
            chunk_overlap = 200
        )
        chunks = splits.split_documents(docs)

        embedding_model = HuggingFaceEmbeddings(
            model_name = "sentence-transformers/all-MiniLM-L6-v2"
        )

        Chroma.from_documents(
            documents = chunks,
            embedding = embedding_model,
            persist_directory = "Chroma_DB"

        )

        return {
            "msg":"PDF uploaded and embedded successfully"
        }
        



    

