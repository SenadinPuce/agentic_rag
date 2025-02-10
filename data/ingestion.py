import os
import tempfile
import logging
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from streamlit.runtime.uploaded_file_manager import UploadedFile


def ingest_docs(file: UploadedFile) -> bool:
    try:
        # Use a temporary directory to automatically manage temp files.
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_file_path = os.path.join(temp_dir, file.name)
            # Write the uploaded file to the temporary file.
            with open(temp_file_path, "wb") as temp_file:
                temp_file.write(file.read())

            # Load the documents using PyPDFLoader.
            loader = PyPDFLoader(temp_file_path)
            raw_documents = loader.load()
            logging.info(f"Loaded {len(raw_documents)} documents")

            # Update the metadata 'source' to the original file name.
            original_file_name = file.name
            for doc in raw_documents:
                doc.metadata["source"] = original_file_name

            # Split the documents into chunks.
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000, chunk_overlap=150
            )
            documents = text_splitter.split_documents(raw_documents)
            logging.info(f"Split into {len(documents)} document chunks")

            # Generate embeddings and store them in Pinecone.
            embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
            PineconeVectorStore.from_documents(
                documents, embeddings, index_name=os.environ["PINECONE_INDEX_NAME"]
            )
            logging.info("Ingested documents into Pinecone successfully")

        return True

    except Exception as e:
        logging.error("Error during document ingestion", exc_info=e)
        raise
