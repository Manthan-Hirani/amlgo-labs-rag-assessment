"""
Embeddings Module
=================
Handles embedding generation using sentence-transformers and
ChromaDB vector database storage/querying.
"""

import os
from pathlib import Path

import chromadb
from chromadb.utils import embedding_functions
from sentence_transformers import SentenceTransformer


# Default embedding model
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
COLLECTION_NAME = "rag_documents"


def get_embedding_function():
    """
    Get the ChromaDB-compatible embedding function using sentence-transformers.
    """
    return embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBEDDING_MODEL
    )


def create_vector_db(
    chunks: list[dict],
    persist_dir: str = "vectordb",
) -> chromadb.Collection:
    """
    Create and populate a ChromaDB collection with document chunks.
    
    Args:
        chunks: List of chunk dicts with 'chunk_id', 'text', 'source_page', 'word_count'
        persist_dir: Directory for persistent ChromaDB storage
    
    Returns:
        The populated ChromaDB collection
    """
    os.makedirs(persist_dir, exist_ok=True)

    # Initialize persistent ChromaDB client
    client = chromadb.PersistentClient(path=persist_dir)

    # Delete existing collection if it exists (for re-indexing)
    try:
        client.delete_collection(name=COLLECTION_NAME)
    except Exception:
        pass

    # Create new collection with embedding function
    ef = get_embedding_function()
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=ef,
        metadata={"hnsw:space": "cosine"}
    )

    # Prepare data for batch insertion
    ids = [str(chunk["chunk_id"]) for chunk in chunks]
    documents = [chunk["text"] for chunk in chunks]
    metadatas = [
        {
            "source_page": chunk["source_page"],
            "word_count": chunk["word_count"],
            "chunk_id": chunk["chunk_id"],
        }
        for chunk in chunks
    ]

    # Add documents in batches to avoid memory issues
    batch_size = 50
    for i in range(0, len(ids), batch_size):
        end = min(i + batch_size, len(ids))
        collection.add(
            ids=ids[i:end],
            documents=documents[i:end],
            metadatas=metadatas[i:end],
        )
        print(f"   Indexed chunks {i+1}-{end} of {len(ids)}")

    print(f"[DONE] Vector DB created with {collection.count()} documents")
    return collection


def get_collection(persist_dir: str = "vectordb") -> chromadb.Collection:
    """
    Load an existing ChromaDB collection.
    """
    client = chromadb.PersistentClient(path=persist_dir)
    ef = get_embedding_function()
    return client.get_collection(
        name=COLLECTION_NAME,
        embedding_function=ef,
    )


def get_collection_count(persist_dir: str = "vectordb") -> int:
    """Get the number of documents in the collection."""
    try:
        collection = get_collection(persist_dir)
        return collection.count()
    except Exception:
        return 0


if __name__ == "__main__":
    from document_loader import load_chunks

    project_root = Path(__file__).parent.parent
    chunks_dir = project_root / "chunks"
    vectordb_dir = project_root / "vectordb"

    print("[LOAD] Loading chunks...")
    chunks = load_chunks(str(chunks_dir))
    print(f"   Loaded {len(chunks)} chunks")

    print("[INDEX] Creating vector database...")
    collection = create_vector_db(chunks, str(vectordb_dir))
    print(f"\n[DONE] Vector DB has {collection.count()} documents.")
