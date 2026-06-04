import streamlit as st
import requests

st.set_page_config(page_title="AI PDF Chatbot")

st.title("📄 AI PDF Chatbot Using RAG")

server_url = "http://127.0.0.1:8000"

if "pdf_uploaded" not in st.session_state:
    st.session_state.pdf_uploaded = False

if "messages" not in st.session_state:
    st.session_state.messages = []

# Upload Section
if not st.session_state.pdf_uploaded:

    uploaded_file = st.file_uploader(
        "Upload PDF",
        type=["pdf"]
    )

    if uploaded_file and st.button("Upload PDF"):

        files = {
            "file": (
                uploaded_file.name,
                uploaded_file,
                "application/pdf"
            )
        }

        with st.spinner("Uploading PDF..."):

            response = requests.post(
                f"{server_url}/uploads",
                files=files
            )

        if response.status_code == 200:

            st.success(
                response.json()["msg"]
            )

            st.session_state.pdf_uploaded = True
            st.rerun()

        else:
            st.error(response.text)

# Chat Section
if st.session_state.pdf_uploaded:

    st.success("PDF Ready for Questions")

    for msg in st.session_state.messages:

        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    question = st.chat_input(
        "Ask anything about your PDF..."
    )

    if question:

        st.session_state.messages.append(
            {
                "role": "user",
                "content": question
            }
        )

        with st.chat_message("user"):
            st.write(question)

        with st.chat_message("assistant"):

            with st.spinner("Thinking..."):

                response = requests.post(
                    f"{server_url}/ask",
                    params={"question": question}
                )

                if response.status_code == 200:

                    answer = response.json()["answer"]

                else:

                    answer = "Backend Error"

            st.write(answer)

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": answer
            }
        )

    if st.button("Exit Session"):

        st.session_state.pdf_uploaded = False
        st.session_state.messages = []

        st.rerun()