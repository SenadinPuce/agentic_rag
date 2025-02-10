import streamlit as st
from backend.workflow import app
from data.ingestion import ingest_docs

st.set_page_config(page_title="Agentic RAG", page_icon="🤖", layout="wide")


def initialize_session_state():
    if "messages" not in st.session_state:
        st.session_state.messages = []


def display_chat_history():
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])


def main_chat():
    st.title("Agentic RAG")
    st.markdown("### Your Personal AI Assistant")

    initialize_session_state()

    display_chat_history()

    with st.sidebar:
        st.header("Upload PDF Document")
        uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
        if uploaded_file:
            with st.spinner("Uploading and processing document..."):
                try:
                    success = ingest_docs(uploaded_file)
                    if success:
                        st.success("Document ingested successfully!")
                    else:
                        st.error("Document ingestion failed!")
                except Exception as e:
                    st.error(f"Error processing document: {e}")

    if prompt := st.chat_input("What would you like to know?"):
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = app.invoke({"question": prompt})

        st.session_state.messages.append(
            {"role": "assistant", "content": response["generation"]}
        )

        st.rerun()


if __name__ == "__main__":
    main_chat()
