# Module 3 — Support Assistant

This module implements a Zepto policy support assistant using local
SentenceTransformer embeddings, ChromaDB retrieval, LangGraph routing,
Pydantic structured responses, and FastAPI.

The graded baseline runs with `MOCK_LLM` unset or set to `1`.
No external LLM API is required in this mode.

## Corpus

The `/docs` directory contains the eight required Zepto policy documents:

- doc_01 — Delivery Policy
- doc_02 — Returns & Refunds
- doc_03 — Membership Tiers
- doc_04 — Order Tracking
- doc_05 — Order Cancellation Policy
- doc_06 — Damaged or Missing Items
- doc_07 — Gift Cards
- doc_08 — Customer Support Hours

Each document is treated as one chunk.

## Architecture

### 1. Ingestion

`load_documents()` in `main.py` loads all eight policy files from `/docs`.

Each document is assigned a chunk ID such as:

`doc_01_chunk_01`

### 2. Embedding

The local SentenceTransformer model:

`sentence-transformers/all-MiniLM-L6-v2`

is used by `embed_texts()` to generate embeddings.

No embedding API key is required.

### 3. Vector Storage

The embeddings are stored in the ChromaDB collection:

`zepto_support_policies`

ChromaDB uses cosine similarity.

The persistent database is stored locally under:

`support_assistant/chroma_db`

### 4. LangGraph Routing

The LangGraph `StateGraph` uses a `TypedDict` state and contains three nodes:

- `classify_intent`
- `retrieve_and_answer`
- `direct_answer`

The flow is:

```text
Incoming query
      |
      v
classify_intent
      |
      +----------------------+
      |                      |
policy_question       general_question
      |                      |
      v                      v
retrieve_and_answer      direct_answer
      |                      |
      +----------+-----------+
                 |
                 v
        Pydantic JSON response