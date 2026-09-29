"""
Ingest schemes_clean.csv into ChromaDB vector store.
Run this ONCE to build the vector database.
"""

import pandas as pd
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
import os
import shutil

# Paths
DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "schemes_clean.csv")
CHROMA_PATH = os.path.join(os.path.dirname(__file__), "..", "chroma_db")


def load_schemes():
    """Load CSV and convert each scheme into a LangChain Document."""
    df = pd.read_csv(DATA_PATH)
    documents = []

    for _, row in df.iterrows():
        # Combine all fields into one rich text block per scheme
        content = f"""Scheme Name: {row['scheme_name']}

Details: {row['details']}

Benefits: {row['benefits']}

Eligibility Criteria: {row['eligibility']}

Application Process: {row['application']}

Documents Required: {row['documents']}"""

        # Store key fields as metadata for filtering and citations
        metadata = {
            "scheme_name": str(row["scheme_name"]),
            "level": str(row.get("level", "")),
            "category": str(row.get("schemeCategory", "")),
            "tags": str(row.get("tags", "")),
            "source_url": str(row.get("source_url", "")),
        }

        documents.append(Document(page_content=content, metadata=metadata))

    print(f"Loaded {len(documents)} schemes from CSV")
    return documents


def chunk_documents(documents):
    """Split long scheme documents into smaller chunks for better retrieval."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", ". ", " "],
    )
    chunks = splitter.split_documents(documents)
    print(f"Split into {len(chunks)} chunks")
    return chunks


def create_vector_store(chunks):
    """Embed chunks and store in ChromaDB."""
    print("Loading embedding model (first time takes a minute)...")
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
    )

    # Remove old DB if exists
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)
        print("Removed old vector store")

    print("Creating vector store (this takes a few minutes)...")
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_PATH,
    )

    print(f"Vector store created at {CHROMA_PATH}")
    print(f"Total vectors stored: {vectorstore._collection.count()}")
    return vectorstore


if __name__ == "__main__":
    documents = load_schemes()
    chunks = chunk_documents(documents)
    create_vector_store(chunks)
    print("\n✅ Ingestion complete! Vector database is ready.")