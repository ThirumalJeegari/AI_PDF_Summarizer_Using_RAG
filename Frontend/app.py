import streamlit as st
import requests
import os

st.set_page_config(page_title="AI PDF Chatbot")

st.title("📄 AI PDF Chatbot Using RAG")

server_url = os.getenv("BACKEND_URL", "http://localhost:8000")

if "pdf_uploaded" not in st.session_state:
    st.session_state.pdf_uploaded = False

if "messages" not in st.session_state:
    st.session_state.messages = []


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

            try:
                response = requests.post(
                    f"{server_url}/uploads",
                    files=files,
                    timeout=300
                )

                if response.status_code == 200:

                    data = response.json()

                    if "msg" in data:
                        st.success(data["msg"])
                        st.session_state.pdf_uploaded = True
                        st.rerun()

                    elif "error" in data:
                        st.error(data["error"])

                    else:
                        st.error(data)

                else:
                    st.error(response.text)

            except Exception as e:
                st.error(f"Backend connection error: {e}")


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

                try:
                    response = requests.post(
                        f"{server_url}/ask",
                        params={"question": question},
                        timeout=300
                    )

                    if response.status_code == 200:

                        data = response.json()
                        answer = data.get("answer", "No answer found")

                    else:
                        answer = "Backend Error"

                except Exception as e:
                    answer = f"Backend connection error: {e}"

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