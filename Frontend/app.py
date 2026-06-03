import streamlit as st
import requests

st.title("AI_PDF_Summarizer_Using_RAG")

serverl_url = "http://127.0.0.1:8000"

File = st.file_uploader("Upload Your File", type = "pdf")
Submit_Button = st.button("Submit")
if Submit_Button:
    res = requests.post(f"{serverl_url}/uploads",files = file)
    st.write(res.json()["msg"])
    


