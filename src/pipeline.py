"""
RAG Pipeline Module
===================
Orchestrates the complete Retrieval-Augmented Generation flow:
Query -> Retrieve -> Generate (with streaming and conversation memory)
"""

from pathlib import Path
from typing import Generator

from src.retriever import retrieve_relevant_chunks, format_context
from src.generator import generate_response, generate_response_stream, DEFAULT_MODEL


# Maximum number of prior conversation turns to include
MAX_HISTORY_TURNS = 3


class RAGPipeline:
    """
    Complete RAG pipeline that combines retrieval, generation,
    and multi-turn conversation memory.
    """

    def __init__(
        self,
        vectordb_dir: str = "vectordb",
        model: str = DEFAULT_MODEL,
        top_k: int = 7,
    ):
        """
        Initialize the RAG pipeline.
        
        Args:
            vectordb_dir: Path to ChromaDB persistent storage
            model: Ollama model name to use for generation
            top_k: Number of chunks to retrieve per query
        """
        self.vectordb_dir = vectordb_dir
        self.model = model
        self.top_k = top_k

    def _prepare_history(
        self, conversation_history: list[dict] | None
    ) -> list[dict] | None:
        """
        Prepare conversation history by trimming to the last
        MAX_HISTORY_TURNS pairs (user + assistant).
        """
        if not conversation_history:
            return None

        # Take only the last N*2 messages (N user-assistant pairs)
        recent = conversation_history[-(MAX_HISTORY_TURNS * 2):]
        if not recent:
            return None
        return recent

    def query(
        self,
        user_query: str,
        conversation_history: list[dict] | None = None,
    ) -> dict:
        """
        Process a query through the full RAG pipeline (non-streaming).
        
        Args:
            user_query: The user's natural language question
            conversation_history: Optional list of prior messages
                Each dict has 'role' and 'content' keys
        
        Returns:
            Dict with 'answer', 'sources', and 'context'
        """
        history = self._prepare_history(conversation_history)

        # Step 1: Retrieve relevant chunks
        sources = retrieve_relevant_chunks(
            user_query,
            top_k=self.top_k,
            persist_dir=self.vectordb_dir,
        )

        # Step 2: Format context
        context = format_context(sources)

        # Step 3: Generate response with conversation history
        answer = generate_response(
            user_query,
            context,
            model=self.model,
            conversation_history=history,
        )

        return {
            "answer": answer,
            "sources": sources,
            "context": context,
        }

    def query_stream(
        self,
        user_query: str,
        conversation_history: list[dict] | None = None,
    ) -> tuple[Generator[str, None, None], list[dict]]:
        """
        Process a query through the RAG pipeline with streaming response
        and conversation memory.
        
        Args:
            user_query: The user's natural language question
            conversation_history: Optional list of prior messages
        
        Returns:
            Tuple of (token_generator, source_chunks)
        """
        history = self._prepare_history(conversation_history)

        # Step 1: Retrieve relevant chunks
        sources = retrieve_relevant_chunks(
            user_query,
            top_k=self.top_k,
            persist_dir=self.vectordb_dir,
        )

        # Step 2: Format context
        context = format_context(sources)

        # Step 3: Create streaming generator with history
        token_stream = generate_response_stream(
            user_query,
            context,
            model=self.model,
            conversation_history=history,
        )

        return token_stream, sources


if __name__ == "__main__":
    project_root = Path(__file__).parent.parent
    vectordb_dir = project_root / "vectordb"

    pipeline = RAGPipeline(vectordb_dir=str(vectordb_dir))

    test_query = "What is this document about?"
    print(f"[QUERY] Query: {test_query}")
    print("=" * 60)

    stream, sources = pipeline.query_stream(test_query)

    print("\n[RESPONSE]:")
    for token in stream:
        print(token, end="", flush=True)

    print(f"\n\n[SOURCES] Sources used: {len(sources)}")
    for i, src in enumerate(sources, 1):
        print(f"   {i}. Page {src['source_page']} (score: {src['relevance_score']})")
