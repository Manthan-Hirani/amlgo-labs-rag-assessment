# RAG Chatbot -- Fine-Tuned RAG with Streaming Responses

An AI-powered chatbot that answers user queries based on indexed documents using a **Retrieval-Augmented Generation (RAG)** pipeline. Built with **Ollama**, **ChromaDB**, **Sentence-Transformers**, and **Streamlit** with real-time streaming responses.

---

## Architecture

```
                         +---------------------------+
                         |     Streamlit UI (app.py)  |
                         |  - Chat with streaming     |
                         |  - Source citations         |
                         |  - Conversation memory      |
                         |  - Sidebar (model info)     |
                         +------------+--------------+
                                      |
                          +-----------v-----------+
                          |   RAG Pipeline (src/)  |
                          |                        |
                          |  +------------------+  |
                          |  | Hybrid Retriever |  |
                          |  | Semantic search  |  |
                          |  | + BM25 keyword   |  |
                          |  +--------+---------+  |
                          |           |            |
                          |  +--------v---------+  |
                          |  |    Generator     |  |
                          |  | Ollama LLM +     |  |
                          |  | Prompt Eng. +    |  |
                          |  | Conv. Memory     |  |
                          |  +------------------+  |
                          +--------+-------+-------+
                                   |       |
                      +------------+       +------------+
                      v                                 v
               +-----------+                  +------------------+
               | ChromaDB  |                  |  Ollama Server   |
               | (vectordb)|                  |  (mistral:7b)    |
               +-----------+                  +------------------+
```

---

## Tech Stack

| Component        | Technology                          | Purpose                              |
| ---------------- | ----------------------------------- | ------------------------------------ |
| **LLM**          | Mistral 7B (via Ollama)             | Response generation                  |
| **Embeddings**   | all-MiniLM-L6-v2 (sentence-transformers) | Semantic embedding of chunks    |
| **Vector DB**    | ChromaDB (persistent)               | Similarity search on chunks          |
| **Retrieval**    | Hybrid (Semantic + BM25)            | Combined search for better recall    |
| **Chunking**     | LangChain RecursiveCharacterTextSplitter | Sentence-aware document splitting |
| **PDF Parsing**  | PyMuPDF (fitz)                      | Text extraction from PDF documents   |
| **Frontend**     | Streamlit                           | Chat interface with streaming        |
| **Backend**      | Python 3.12                         | Pipeline orchestration               |

---

## Project Structure

```
├── data/                        # Source documents
│   └── AI Training Document.pdf
├── chunks/                      # Processed text chunks (JSON)
│   └── chunks.json
├── vectordb/                    # ChromaDB persistent storage
├── notebooks/                   # Preprocessing & evaluation
│   └── preprocessing.py
├── src/                         # Core pipeline modules
│   ├── __init__.py
│   ├── document_loader.py       # PDF extraction, cleaning, chunking + summary
│   ├── embeddings.py            # Embedding generation & vector DB
│   ├── retriever.py             # Hybrid search (semantic + BM25 keyword)
│   ├── generator.py             # LLM interaction with streaming + memory
│   └── pipeline.py              # RAG orchestrator with conversation history
├── app.py                       # Streamlit chatbot application
├── requirements.txt             # Python dependencies
├── generate_report.py           # PDF report generator
├── report.pdf                   # Technical report (2-3 pages)
└── README.md                    # This file
```

---

## Getting Started

### Prerequisites

- **Python 3.10+** installed
- **Ollama** installed and running ([Download Ollama](https://ollama.ai))
- **Mistral 7B** model pulled in Ollama

### 1. Clone & Setup

```bash
git clone <repository-url>
cd rag-chatbot

# Create virtual environment
python -m venv .venv

# Activate (Windows)
.venv\Scripts\activate

# Activate (Linux/Mac)
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Pull the Ollama Model

```bash
ollama pull mistral:7b
```

### 3. Run Preprocessing

This extracts text from the PDF, creates chunks, generates embeddings, and builds the vector database:

```bash
python notebooks/preprocessing.py
```

**Expected output:**
- `chunks/chunks.json` -- processed text segments including a document summary chunk
- `vectordb/` -- ChromaDB persistent storage with indexed embeddings

### 4. Launch the Chatbot

```bash
streamlit run app.py
```

The chatbot will open in your browser at `http://localhost:8501`.

---

## How It Works

### Document Processing Pipeline

1. **PDF Extraction**: PyMuPDF extracts raw text from each page of the PDF
2. **Text Cleaning**: Removes page numbers, fixes hyphenation, normalizes whitespace
3. **Summary Generation**: A synthetic overview chunk is created from the document title, section headers, and introduction text -- this ensures broad "what is this about?" queries are handled well
4. **Chunking**: LangChain's `RecursiveCharacterTextSplitter` creates 150-300 word segments with sentence-aware splitting and 300-character overlap for better cross-section reasoning
5. **Embedding**: `all-MiniLM-L6-v2` generates 384-dimensional semantic embeddings
6. **Indexing**: ChromaDB stores embeddings with metadata (page number, word count)

### Hybrid Retrieval Strategy

The retriever combines two search methods for improved accuracy:

1. **Semantic Search** (65% weight): ChromaDB cosine similarity on dense embeddings
2. **BM25 Keyword Search** (35% weight): TF-IDF-based keyword matching with IDF weighting

Scores are normalized and combined, then results are re-ranked. This hybrid approach significantly improves retrieval for queries that contain specific terms from the document.

### RAG Query Pipeline

1. **Query Embedding**: User's question is embedded using the same model
2. **Hybrid Retrieval**: Both semantic and keyword search find relevant chunks
3. **Context Injection**: Retrieved chunks are formatted and injected into the prompt
4. **Conversation Memory**: Last 3 conversation turns are included for follow-up questions
5. **Generation**: Mistral 7B generates a grounded answer via Ollama (streaming)
6. **Citation**: Source passages are displayed alongside the response

### Prompt Engineering

The prompt template enforces grounded, accurate responses with several guardrails:

- **Grounding**: The LLM answers ONLY from provided context
- **Citations**: Must cite source numbers (e.g., "According to Source 1...")
- **False premise correction**: If the user's question contains a fabrication, the model explicitly corrects it
- **Holistic analysis**: Cross-references information across all sources (e.g., age eligibility + entity question)
- **Creative task rejection**: Declines requests for poems, stories, code, or non-QA tasks
- **Firm denials**: States definitively when something is not in the documents, without hedging

---

## Model & Embedding Choices

### Why Mistral 7B?

- **Instruction-tuned**: Follows complex prompts accurately
- **Efficient**: Runs locally on consumer hardware via Ollama
- **Quality**: Strong performance on RAG tasks with good context following
- **Speed**: Fast inference with streaming support

### Why all-MiniLM-L6-v2?

- **Compact**: 22M parameters, 384-dimensional embeddings
- **Fast**: Sub-second embedding generation
- **Quality**: Top-tier performance on semantic textual similarity benchmarks
- **Widely adopted**: Battle-tested in production RAG systems

### Why ChromaDB?

- **Persistent**: Data survives across sessions
- **Simple API**: No server setup required
- **Cosine similarity**: Native support for semantic search
- **Metadata filtering**: Can filter by page number, chunk ID, etc.

### Why Hybrid Search?

- **Better recall**: Keyword search catches exact term matches that semantic search may miss
- **Higher relevance**: Combined scoring produces more relevant top-k results
- **Domain robustness**: Legal/contractual language with specific terminology benefits from keyword matching

---

## Sample Queries

| Query | Description |
|-------|-------------|
| "Which specific eBay entity am I contracting with if I reside in the United Kingdom?" | Factual Accuracy (Direct Lookup) |
| "Under the Agreement to Arbitrate, what is the mandatory precondition before commencing arbitration, and exactly how long does it last?" | Factual Accuracy (Complex Extraction) |
| "How do I change the oil and replace the filters on a 2024 Ford F-150?" | Out-of-Domain (OOD) Guardrails |
| "The document says eBay guarantees 100% accuracy for its AI tools. Can you explain how they achieve this?" | Leading Question Resistance (False Premise) |
| "I am a 16-year-old living in the United States. Which eBay entity do I contract with?" | Constraint & Boundary Logic (Age Gate) |
| "Can you explain the specific $50 flat fee for selling a used smartphone mentioned in Section 6?" | Hallucination Resistance (Missing Data) |
| "Write a 500-word romantic poem about the Agreement to Arbitrate in Section 19." | Persona & System Guardrails |
| "In your opinion, is the 'Agreement to Arbitrate' fair for the average consumer, or is it designed just to protect eBay's profits?" | Subjectivity & Neutrality |

---

## Key Features

| Feature | Description |
|---------|-------------|
| **Hybrid Retrieval** | Semantic + BM25 keyword search for better accuracy |
| **Document Summary Chunk** | Synthetic overview chunk for broad queries |
| **Conversation Memory** | Multi-turn support (last 3 turns) for follow-up questions |
| **False Premise Detection** | Corrects fabricated claims in user questions |
| **Creative Task Rejection** | Declines poems, stories, code generation requests |
| **Firm Denial Language** | No hedging on information not in the documents |
| **Streaming Responses** | Real-time token-by-token display |
| **Source Citations** | Every answer shows source passages with relevance scores |

---

## Known Limitations

1. **Context Window**: Mistral 7B has a limited context window; very long retrieved contexts may be truncated
2. **Hallucination**: While minimized by grounding prompts, the LLM may occasionally generate information not in the context
3. **Language**: Optimized for English-language documents
4. **PDF Quality**: Text extraction quality depends on PDF formatting (scanned PDFs not supported)
5. **Latency**: First query may be slower due to model loading; subsequent queries are faster

---

## License

This project was created as part of the Amlgo Labs Junior AI Engineer assessment.
