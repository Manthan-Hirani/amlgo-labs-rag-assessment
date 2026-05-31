"""
Retriever Module
================
Performs hybrid search (semantic + keyword) against the ChromaDB
vector database with re-ranking for improved retrieval quality.
"""

import math
import re
from collections import Counter
from pathlib import Path

from src.embeddings import get_collection


# ─── BM25 Keyword Scoring ────────────────────────────────────────────

def _tokenize(text: str) -> list[str]:
    """Simple tokenizer: lowercase, split on non-alphanumeric."""
    return re.findall(r'[a-z0-9]+', text.lower())


def _compute_idf(corpus_tokens: list[list[str]]) -> dict[str, float]:
    """Compute IDF scores for terms across the corpus."""
    n_docs = len(corpus_tokens)
    df = Counter()
    for doc_tokens in corpus_tokens:
        unique_tokens = set(doc_tokens)
        for token in unique_tokens:
            df[token] += 1

    idf = {}
    for token, freq in df.items():
        idf[token] = math.log((n_docs - freq + 0.5) / (freq + 0.5) + 1)
    return idf


def _bm25_score(
    query_tokens: list[str],
    doc_tokens: list[str],
    idf: dict[str, float],
    avg_dl: float,
    k1: float = 1.5,
    b: float = 0.75,
) -> float:
    """Compute BM25 score for a single document."""
    dl = len(doc_tokens)
    tf_counter = Counter(doc_tokens)
    score = 0.0
    for token in query_tokens:
        if token not in idf:
            continue
        tf = tf_counter.get(token, 0)
        numerator = idf[token] * tf * (k1 + 1)
        denominator = tf + k1 * (1 - b + b * dl / avg_dl)
        score += numerator / denominator
    return score


def keyword_search(
    query: str,
    documents: list[dict],
    top_k: int = 15,
) -> list[tuple[int, float]]:
    """
    Perform BM25 keyword search across document chunks.
    
    Args:
        query: User's query string
        documents: List of chunk dicts with 'text' and 'chunk_id'
        top_k: Number of results to return
    
    Returns:
        List of (chunk_index, bm25_score) tuples, sorted by score desc
    """
    query_tokens = _tokenize(query)
    if not query_tokens:
        return []

    corpus_tokens = [_tokenize(doc["text"]) for doc in documents]
    avg_dl = sum(len(t) for t in corpus_tokens) / len(corpus_tokens) if corpus_tokens else 1

    idf = _compute_idf(corpus_tokens)

    scored = []
    for i, doc_tokens in enumerate(corpus_tokens):
        score = _bm25_score(query_tokens, doc_tokens, idf, avg_dl)
        if score > 0:
            scored.append((i, score))

    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:top_k]


# ─── Hybrid Retrieval ────────────────────────────────────────────────

def retrieve_relevant_chunks(
    query: str,
    top_k: int = 7,
    persist_dir: str = "vectordb",
    semantic_weight: float = 0.65,
    keyword_weight: float = 0.35,
) -> list[dict]:
    """
    Retrieve the top-k most relevant chunks using hybrid search:
    combines semantic similarity (ChromaDB) with BM25 keyword matching.
    
    Args:
        query: The user's natural language query
        top_k: Number of chunks to return
        persist_dir: Path to the ChromaDB directory
        semantic_weight: Weight for semantic similarity score (0-1)
        keyword_weight: Weight for keyword BM25 score (0-1)
    
    Returns:
        List of dicts with 'text', 'source_page', 'word_count',
        'chunk_id', and 'relevance_score'
    """
    collection = get_collection(persist_dir)
    total_docs = collection.count()

    # Step 1: Get a broad set of semantic results (2x top_k)
    n_semantic = min(top_k * 3, total_docs)
    semantic_results = collection.query(
        query_texts=[query],
        n_results=n_semantic,
        include=["documents", "metadatas", "distances"],
    )

    # Build a lookup of all retrieved chunks by chunk_id
    chunk_map = {}
    if semantic_results and semantic_results["documents"]:
        for i, doc in enumerate(semantic_results["documents"][0]):
            metadata = semantic_results["metadatas"][0][i]
            distance = semantic_results["distances"][0][i]
            similarity = max(0, 1 - distance)  # Cosine similarity

            chunk_id = metadata.get("chunk_id", i)
            chunk_map[chunk_id] = {
                "text": doc,
                "source_page": metadata.get("source_page", "N/A"),
                "word_count": metadata.get("word_count", 0),
                "chunk_id": chunk_id,
                "semantic_score": similarity,
                "keyword_score": 0.0,
            }

    # Step 2: Run keyword (BM25) search on the same chunks
    docs_list = list(chunk_map.values())
    if docs_list:
        bm25_results = keyword_search(query, docs_list, top_k=n_semantic)

        # Normalize BM25 scores to 0-1 range
        if bm25_results:
            max_bm25 = max(score for _, score in bm25_results)
            if max_bm25 > 0:
                for idx, score in bm25_results:
                    chunk_id = docs_list[idx]["chunk_id"]
                    chunk_map[chunk_id]["keyword_score"] = score / max_bm25

    # Step 3: Compute hybrid scores and rank
    for chunk in chunk_map.values():
        chunk["relevance_score"] = round(
            semantic_weight * chunk["semantic_score"]
            + keyword_weight * chunk["keyword_score"],
            4,
        )

    # Sort by hybrid score and return top_k
    ranked = sorted(chunk_map.values(), key=lambda x: x["relevance_score"], reverse=True)

    # Clean up internal scores before returning
    results = []
    for chunk in ranked[:top_k]:
        results.append({
            "text": chunk["text"],
            "source_page": chunk["source_page"],
            "word_count": chunk["word_count"],
            "chunk_id": chunk["chunk_id"],
            "relevance_score": chunk["relevance_score"],
        })

    return results


def format_context(chunks: list[dict]) -> str:
    """
    Format retrieved chunks into a single context string
    for injection into the LLM prompt.
    """
    context_parts = []
    for i, chunk in enumerate(chunks, 1):
        page_label = "Overview" if chunk["source_page"] == 0 else f"Page {chunk['source_page']}"
        context_parts.append(
            f"[Source {i} | {page_label}]\n{chunk['text']}"
        )
    return "\n\n---\n\n".join(context_parts)


if __name__ == "__main__":
    project_root = Path(__file__).parent.parent
    vectordb_dir = project_root / "vectordb"

    test_queries = [
        "What is this document about?",
        "What guidelines or rules are mentioned?",
        "What is the refund policy?",
    ]

    for test_query in test_queries:
        print(f"\n[QUERY] Query: {test_query}")
        print("=" * 60)

        results = retrieve_relevant_chunks(test_query, top_k=3, persist_dir=str(vectordb_dir))

        for i, chunk in enumerate(results, 1):
            print(f"  [RESULT] {i}. (Score: {chunk['relevance_score']:.4f}, Page {chunk['source_page']})")
            print(f"           {chunk['text'][:150]}...")
        print()
