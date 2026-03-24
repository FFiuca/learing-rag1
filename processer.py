from dotenv import load_dotenv
import os
from pathlib import Path
from datetime import datetime
from typing import List, Optional

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores.faiss import FAISS
from transformers import pipeline

from db import get_mongo_db

# Load environment variables
load_dotenv()

# Initialize models
EMBEDDING_MODEL = "nomic-embed-text"
CLASSIFIER_MODEL = "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli"

# Lazy load classifier
_classifier = None

def get_classifier():
    """Lazy load zero-shot classifier."""
    global _classifier
    if _classifier is None:
        print("Loading classifier model...")
        _classifier = pipeline("zero-shot-classification", model=CLASSIFIER_MODEL)
    return _classifier


def load_text_file(file_path: str) -> List:
    """Load text file using TextLoader."""
    loader = TextLoader(file_path)
    return loader.load()


def chunk_text(documents: List, chunk_size: int = 1000, chunk_overlap: int = 200) -> List:
    """
    Split documents into chunks.

    Args:
        documents: List of document objects
        chunk_size: Size of each chunk
        chunk_overlap: Overlap between chunks

    Returns:
        List of chunked documents
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    all_chunks = []
    for doc in documents:
        chunks = splitter.split_documents([doc])
        all_chunks.extend(chunks)
    return all_chunks


def embed_chunks(chunks: List) -> OllamaEmbeddings:
    """Create embeddings for chunks."""
    return OllamaEmbeddings(model=EMBEDDING_MODEL)


def create_vector_store(chunks: List, embeddings: OllamaEmbeddings, index_name: str = "faiss_index") -> FAISS:
    """
    Create and save FAISS vector store.

    Args:
        chunks: List of document chunks
        embeddings: Embeddings model
        index_name: Name of the FAISS index

    Returns:
        FAISS vector store
    """
    vector_store = FAISS.from_documents(chunks, embeddings)
    vector_store.save_local(index_name)
    print(f"Vector store saved to {index_name}")
    return vector_store


def categorize_query(text: str) -> List[str]:
    """
    Classify text into active categories from MongoDB.

    Args:
        text: Text to categorize

    Returns:
        List of category names with score > 0.1
    """
    # Get active categories from MongoDB
    db = get_mongo_db()
    categories_data = db["category"].find({"is_active": True})
    category_names = [cat["name"] for cat in categories_data]

    if not category_names:
        print("No categories found in database")
        return []

    print(f"Available categories: {category_names}")

    # Classify text
    classifier = get_classifier()
    classification = classifier(text, candidate_labels=category_names)
    print(f"Classification result: {classification}")

    # Filter categories with score > 0.1
    categories_result = [
        classification["labels"][i]
        for i in range(len(classification["labels"]))
        if classification["scores"][i] > 0.1
    ]

    return categories_result


def save_to_db(text: str, categories: List[str]) -> None:
    """
    Save document text and categories to MongoDB.

    Args:
        text: Document text
        categories: List of categories
    """
    db = get_mongo_db()
    db["metadata"].insert_one({
        "text": text,
        "categories": categories,
        "created_at": datetime.utcnow(),
        "is_active": True
    })
    print("Data saved to MongoDB")


def process_document(file_path: str) -> None:
    """
    Process a single document: load, categorize, save metadata, chunk, embed.

    Args:
        file_path: Path to the document file
    """
    print(f"Processing file: {file_path}")

    # Load document
    document = load_text_file(file_path)
    content = document[0].page_content

    # Categorize and save
    categories = categorize_query(content)
    print(f"Categories: {categories}")
    save_to_db(content, categories)

    # Chunk and embed
    chunks = chunk_text(document)
    print(f"Created {len(chunks)} chunks")

    embeddings = embed_chunks(chunks)
    vector_store = create_vector_store(chunks, embeddings)
    print(f"Vector store created with {len(chunks)} documents\n")


def process_documents(file_paths: List[str]) -> None:
    """
    Process multiple documents.

    Args:
        file_paths: List of file paths to process
    """
    print(f"Processing {len(file_paths)} files\n")

    # Load all documents
    documents = [load_text_file(path) for path in file_paths]

    # Process each document
    for doc in documents:
        content = doc[0].page_content
        categories = categorize_query(content)
        print(f"Categories: {categories}")
        save_to_db(content, categories)

    # Chunk all documents together
    all_chunks = chunk_text(documents)
    print(f"Created {len(all_chunks)} total chunks")

    # Create vector store
    embeddings = embed_chunks(all_chunks)
    vector_store = create_vector_store(all_chunks, embeddings)
    print(f"Vector store created with {len(all_chunks)} documents")


def main():
    """Example usage: process all .txt files in data directory."""
    print("Initializing processor...")

    # Get all text files in data directory
    data_dir = Path.cwd() / "data"
    file_paths = [str(f) for f in data_dir.glob("*.txt")]

    if not file_paths:
        print(f"No text files found in {data_dir}")
        return

    # Process documents
    process_documents(file_paths)


if __name__ == "__main__":
    main()
