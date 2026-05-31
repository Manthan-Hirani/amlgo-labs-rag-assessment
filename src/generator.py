"""
Generator Module
================
Handles LLM interaction via Ollama for generating grounded
responses with streaming support. Includes conversation
history for multi-turn understanding.
"""

from typing import Generator

import ollama


# Default model configuration
DEFAULT_MODEL = "mistral:7b"


SYSTEM_PROMPT = """You are a precise, factual AI assistant that answers questions based ONLY on provided document context.

STRICT RULES:

1. GROUNDING: Answer ONLY using the provided context passages. NEVER use external knowledge or make assumptions beyond what is stated in the context.

2. CITATIONS: ALWAYS cite which source passages support your answer (e.g., "According to Source 1, Page 3..."). Every factual claim must reference a source.

3. INSUFFICIENT INFORMATION: If the context does not contain enough information to fully answer the question, state clearly: "The provided documents do not contain information to answer this question." Do NOT speculate or hedge.

4. FALSE PREMISES: If the user's question contains a false premise or fabricated claim (e.g., claiming the document says something it does not), you MUST explicitly correct the premise. State firmly: "This is not stated in the provided documents. In fact, [correct information from context]." Do NOT leave room for doubt about fabricated claims.

5. HOLISTIC ANALYSIS: Consider ALL retrieved context passages together. Cross-reference information across sources. For example:
   - If a user mentions their age and the document has age/eligibility requirements, address BOTH the literal question AND any relevant eligibility conditions.
   - If a user asks about an entity but context also mentions restrictions or prerequisites, include those in your answer.

6. QUESTION-ANSWERING ONLY: You are strictly a question-answering assistant. Do NOT:
   - Write poems, stories, essays, songs, or any creative/fictional content
   - Generate code, scripts, or technical artifacts
   - Perform tasks unrelated to answering factual questions about the documents
   - Engage in role-play or hypothetical scenarios
   If asked to do any of the above, respond: "I can only answer factual questions about the provided documents. I cannot generate creative content, write code, or perform tasks outside of document-based Q&A."

7. FIRM DENIALS: When something is not mentioned in the documents, state this definitively. Do NOT hedge with phrases like "it is impossible to determine" or "without more information." Instead say: "This is not mentioned anywhere in the provided documents."

8. STRUCTURE: Keep answers clear, well-structured, and concise. Use bullet points for multi-part answers.

9. AMBIGUITY: If the question is ambiguous, interpret it in the most reasonable way based on context and state your interpretation."""


def build_prompt(
    query: str,
    context: str,
    conversation_history: list[dict] | None = None,
) -> str:
    """
    Build the prompt that combines retrieved context, conversation
    history, and the user query.
    
    Args:
        query: The user's question
        context: Formatted context string from retrieved chunks
        conversation_history: Optional list of prior conversation turns
            Each dict has 'role' ('user'/'assistant') and 'content'
    
    Returns:
        The complete prompt string
    """
    # Build conversation history section if available
    history_section = ""
    if conversation_history:
        history_parts = []
        for turn in conversation_history:
            role_label = "USER" if turn["role"] == "user" else "ASSISTANT"
            # Truncate long prior answers to save context window
            content = turn["content"]
            if len(content) > 500:
                content = content[:500] + "..."
            history_parts.append(f"{role_label}: {content}")
        history_section = (
            "\n=== CONVERSATION HISTORY (for context) ===\n"
            + "\n\n".join(history_parts)
            + "\n=== END HISTORY ===\n\n"
        )

    return f"""{history_section}Based on the following document passages, answer the user's question.

=== DOCUMENT CONTEXT ===
{context}
=== END CONTEXT ===

USER QUESTION: {query}

Provide a thorough, accurate answer based solely on the above context. Reference specific sources when possible. If the user's question contains a false claim, correct it. Consider all context holistically."""


def generate_response(
    query: str,
    context: str,
    model: str = DEFAULT_MODEL,
    conversation_history: list[dict] | None = None,
) -> str:
    """
    Generate a complete (non-streaming) response.
    
    Args:
        query: User's question
        context: Retrieved document context
        model: Ollama model to use
        conversation_history: Optional prior conversation turns
    
    Returns:
        The complete response string
    """
    prompt = build_prompt(query, context, conversation_history)

    response = ollama.chat(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        options={"temperature": 0.0}
    )
    return response["message"]["content"]


def generate_response_stream(
    query: str,
    context: str,
    model: str = DEFAULT_MODEL,
    conversation_history: list[dict] | None = None,
) -> Generator[str, None, None]:
    """
    Generate a streaming response, yielding tokens as they arrive.
    
    Args:
        query: User's question
        context: Retrieved document context
        model: Ollama model to use
        conversation_history: Optional prior conversation turns
    
    Yields:
        Individual tokens/chunks of the response
    """
    prompt = build_prompt(query, context, conversation_history)

    stream = ollama.chat(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        stream=True,
        options={"temperature": 0.0}
    )

    for chunk in stream:
        token = chunk["message"]["content"]
        if token:
            yield token


if __name__ == "__main__":
    # Quick test
    test_context = "This is a test document about AI training. It covers machine learning basics."
    test_query = "What does the document cover?"

    print("[TEST] Testing streaming response...")
    print("=" * 60)
    for token in generate_response_stream(test_query, test_context):
        print(token, end="", flush=True)
    print("\n" + "=" * 60)
    print("[DONE] Streaming test complete!")
