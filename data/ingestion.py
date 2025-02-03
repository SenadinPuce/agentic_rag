import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore


def ingest_docs():
    # Load the documents

    loader = PyPDFLoader("data\\Tehnoloski procesi_A4.pdf")
    raw_documents = loader.load()
    print(f"Loaded {len(raw_documents)} documents")

    # Split the documents into sentences

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    documents = text_splitter.split_documents(raw_documents)
    print(f"Split {len(documents)} documents into chunks")

    # Embed documents

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    PineconeVectorStore.from_documents(
        documents, embeddings, index_name=os.environ["PINECONE_INDEX_NAME"]
    )

    print("Ingested documents into Pinecone")


if __name__ == "__main__":
    ingest_docs()
