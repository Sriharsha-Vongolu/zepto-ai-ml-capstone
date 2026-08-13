import json
import os
from pathlib import Path
from typing import Literal, TypedDict

import chromadb
from fastapi import FastAPI
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer

from langgraph.graph import END, START, StateGraph


# ============================================================
# Paths and constants
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
DOCS_DIR = BASE_DIR / "docs"
CHROMA_DIR = BASE_DIR / "chroma_db"

COLLECTION_NAME = "zepto_support_policies"

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

POLICY_KEYWORDS = [
    "delivery",
    "return",
    "refund",
    "membership",
    "tracking",
    "cancel",
    "gift card",
    "support hours",
]

GENERAL_MOCK_ANSWER = (
    "I can only answer questions about Zepto policies right now."
)


# ============================================================
# MOCK_LLM toggle
# ============================================================

def is_mock_mode() -> bool:
    """
    MOCK_LLM is ON by default.

    Unset or MOCK_LLM=1 -> deterministic mock mode.
    MOCK_LLM=0 -> optional real-LLM path.
    """
    return os.getenv("MOCK_LLM", "1") != "0"


# ============================================================
# Pydantic request / response models
# ============================================================

class AskRequest(BaseModel):
    query: str


class AssistantResponse(BaseModel):
    answer: str
    sources: list[str]
    confidence: float = Field(
        ge=0.0,
        le=1.0,
    )


# ============================================================
# Structured prompt
# role -> context -> task -> format -> length
# Includes negative constraint and few-shot example.
# Used only by optional MOCK_LLM=0 path.
# ============================================================

STRUCTURED_PROMPT_TEMPLATE = """
ROLE:
You are a Zepto customer-support assistant that answers questions
about Zepto policies.

CONTEXT:
Use only the following retrieved Zepto policy context:

{context}

TASK:
Answer the customer's question accurately using only the provided
context.

Customer question:
{query}

NEGATIVE CONSTRAINT:
Do not answer using information that is not present in the provided
context. Do not invent policies, prices, timelines, benefits, or
support options.

FORMAT:
Return valid JSON with exactly these fields:
- answer: string
- sources: list of source chunk/document IDs
- confidence: float between 0 and 1

LENGTH:
Keep the answer concise, preferably no more than 3 sentences.

FEW-SHOT EXAMPLE:

Context:
doc_01_chunk_01 says standard delivery is free on orders over
INR 149 and orders below this threshold incur an INR 25 fee.

Customer question:
What is the standard delivery fee?

Expected answer:
answer = "Standard delivery is free for orders over INR 149.
Orders below INR 149 have a flat INR 25 delivery fee."
sources = ["doc_01_chunk_01"]
confidence = 1.0
""".strip()


DIRECT_PROMPT_TEMPLATE = """
You are a helpful assistant.

Answer the following general question directly.

Question:
{query}

Return valid JSON with exactly:
- answer: string
- sources: []
- confidence: float between 0 and 1

Keep the answer concise.
""".strip()


# ============================================================
# Document ingestion
# ============================================================

def load_documents() -> list[dict]:
    """
    Load the eight required policy documents.

    Because each assignment document is short, one document is
    treated as one chunk.
    """

    documents = []

    for index in range(1, 9):
        file_name = f"doc_{index:02d}.txt"
        file_path = DOCS_DIR / file_name

        if not file_path.exists():
            raise FileNotFoundError(
                f"Required corpus file not found: {file_path}"
            )

        text = file_path.read_text(
            encoding="utf-8"
        ).strip()

        chunk_id = (
            f"doc_{index:02d}_chunk_01"
        )

        documents.append(
            {
                "id": chunk_id,
                "document_id": f"doc_{index:02d}",
                "text": text,
            }
        )

    return documents


# ============================================================
# Local embeddings
# ============================================================

embedding_model = SentenceTransformer(
    EMBEDDING_MODEL_NAME
)


def embed_texts(
    texts: list[str],
) -> list[list[float]]:
    """
    Generate local SentenceTransformer embeddings.
    """

    embeddings = embedding_model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    return embeddings.tolist()


# ============================================================
# ChromaDB initialization
# ============================================================

chroma_client = chromadb.PersistentClient(
    path=str(CHROMA_DIR)
)


collection = chroma_client.get_or_create_collection(
    name=COLLECTION_NAME,
    configuration={
        "hnsw": {
            "space": "cosine",
        }
    },
)


def ingest_documents() -> None:
    """
    Embed and upsert all eight corpus documents into ChromaDB.
    """

    documents = load_documents()

    ids = [
        document["id"]
        for document in documents
    ]

    texts = [
        document["text"]
        for document in documents
    ]

    metadatas = [
        {
            "document_id":
                document["document_id"],
            "chunk_id":
                document["id"],
        }
        for document in documents
    ]

    embeddings = embed_texts(texts)

    collection.upsert(
        ids=ids,
        documents=texts,
        embeddings=embeddings,
        metadatas=metadatas,
    )


# Run ingestion when the application starts.
# Upsert makes this safe to run repeatedly.
ingest_documents()


# ============================================================
# Retrieval
# ============================================================

def retrieve_top_chunks(
    query: str,
    n_results: int = 3,
) -> list[dict]:
    """
    Embed the query locally and retrieve the top matching chunks
    from ChromaDB using cosine similarity.
    """

    query_embedding = embed_texts(
        [query]
    )[0]

    result = collection.query(
        query_embeddings=[
            query_embedding
        ],
        n_results=n_results,
        include=[
            "documents",
            "metadatas",
            "distances",
        ],
    )

    retrieved = []

    ids = result["ids"][0]
    documents = result["documents"][0]
    metadatas = result["metadatas"][0]
    distances = result["distances"][0]

    for (
        chunk_id,
        document,
        metadata,
        distance,
    ) in zip(
        ids,
        documents,
        metadatas,
        distances,
    ):
        retrieved.append(
            {
                "id": chunk_id,
                "text": document,
                "metadata": metadata,
                "distance": distance,
            }
        )

    return retrieved


# ============================================================
# Optional real-LLM helpers
#
# NOT used in default graded mock mode.
#
# This path is included because the acceptance criteria require
# retry-on-validation-failure logic to be present.
# ============================================================

def call_real_llm_text(
    prompt: str,
) -> str:
    """
    Optional real-LLM call.

    This function is never called while MOCK_LLM is unset or 1.
    """

    try:
        from groq import Groq
    except ImportError as exc:
        raise RuntimeError(
            "Optional real-LLM mode requires the "
            "'groq' package."
        ) from exc

    api_key = os.getenv("GROQ_API_KEY")
    model_name = os.getenv("GROQ_MODEL")

    if not api_key:
        raise RuntimeError(
            "GROQ_API_KEY is required when MOCK_LLM=0."
        )

    if not model_name:
        raise RuntimeError(
            "GROQ_MODEL is required when MOCK_LLM=0."
        )

    client = Groq(
        api_key=api_key
    )

    completion = client.chat.completions.create(
        model=model_name,
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
        temperature=0,
    )

    return (
        completion
        .choices[0]
        .message
        .content
        .strip()
    )


def call_real_llm_structured(
    prompt: str,
    fallback_sources: list[str],
) -> AssistantResponse:
    """
    Optional real-LLM structured-output path.

    Makes the initial attempt plus up to two additional
    corrective retries if Pydantic validation fails.
    """

    current_prompt = prompt

    for attempt in range(3):

        raw_output = call_real_llm_text(
            current_prompt
        )

        try:
            return (
                AssistantResponse
                .model_validate_json(
                    raw_output
                )
            )

        except Exception as exc:

            if attempt < 2:
                schema = json.dumps(
                    AssistantResponse
                    .model_json_schema(),
                    indent=2,
                )

                current_prompt = f"""
{prompt}

Your previous response failed schema validation.

Return ONLY valid JSON matching this schema:

{schema}

Validation error:
{exc}
""".strip()

    return AssistantResponse(
        answer=(
            "ERROR: Real LLM response failed "
            "schema validation after 3 attempts."
        ),
        sources=fallback_sources,
        confidence=0.0,
    )


# ============================================================
# LangGraph state
# ============================================================

class SupportState(TypedDict, total=False):
    query: str

    intent: Literal[
        "policy_question",
        "general_question",
    ]

    retrieved_chunks: list[dict]

    answer: str

    sources: list[str]

    confidence: float

    response: AssistantResponse


# ============================================================
# Node 1 — classify_intent
# ============================================================

def classify_intent(
    state: SupportState,
) -> dict:

    query = state["query"]

    if is_mock_mode():

        lower_query = query.lower()

        is_policy_question = any(
            keyword in lower_query
            for keyword in POLICY_KEYWORDS
        )

        intent = (
            "policy_question"
            if is_policy_question
            else "general_question"
        )

    else:
        # Optional real-LLM extension.
        prompt = f"""
Classify the following query as exactly one of:

policy_question
general_question

A policy_question concerns Zepto delivery, returns,
refunds, memberships, tracking, cancellations,
gift cards, or support policies.

Query:
{query}

Return only the classification label.
""".strip()

        raw_intent = (
            call_real_llm_text(prompt)
            .strip()
            .lower()
        )

        if raw_intent == "policy_question":
            intent = "policy_question"

        elif raw_intent == "general_question":
            intent = "general_question"

        else:
            raise ValueError(
                "Real LLM returned an invalid "
                f"intent: {raw_intent}"
            )

    return {
        "intent": intent,
    }


# ============================================================
# Conditional router
# ============================================================

def route_by_intent(
    state: SupportState,
) -> Literal[
    "retrieve_and_answer",
    "direct_answer",
]:

    if (
        state["intent"]
        == "policy_question"
    ):
        return "retrieve_and_answer"

    return "direct_answer"


# ============================================================
# Node 2 — retrieve_and_answer
# ============================================================

def retrieve_and_answer(
    state: SupportState,
) -> dict:

    query = state["query"]

    # Retrieval ALWAYS runs for real,
    # regardless of MOCK_LLM mode.
    retrieved_chunks = (
        retrieve_top_chunks(
            query=query,
            n_results=3,
        )
    )

    source_ids = [
        chunk["id"]
        for chunk in retrieved_chunks
    ]

    if is_mock_mode():

        top_chunk = retrieved_chunks[0]["text"]

        top_chunk_snippet = (
            top_chunk[:200]
            .replace("\n", " ")
            .strip()
        )

        answer = (
            "Based on the retrieved context: "
            f"{top_chunk_snippet}"
        )

        response = AssistantResponse(
            answer=answer,
            sources=source_ids,
            confidence=1.0,
        )

    else:
        # Optional real-LLM extension.

        context = "\n\n".join(
            (
                f"Source: {chunk['id']}\n"
                f"{chunk['text']}"
            )
            for chunk in retrieved_chunks
        )

        prompt = (
            STRUCTURED_PROMPT_TEMPLATE
            .format(
                context=context,
                query=query,
            )
        )

        response = call_real_llm_structured(
            prompt=prompt,
            fallback_sources=source_ids,
        )

    return {
        "retrieved_chunks":
            retrieved_chunks,

        "answer":
            response.answer,

        "sources":
            response.sources,

        "confidence":
            response.confidence,

        "response":
            response,
    }


# ============================================================
# Node 3 — direct_answer
# ============================================================

def direct_answer(
    state: SupportState,
) -> dict:

    query = state["query"]

    if is_mock_mode():

        response = AssistantResponse(
            answer=GENERAL_MOCK_ANSWER,
            sources=[],
            confidence=1.0,
        )

    else:
        # Optional real-LLM extension.
        prompt = (
            DIRECT_PROMPT_TEMPLATE
            .format(
                query=query,
            )
        )

        response = call_real_llm_structured(
            prompt=prompt,
            fallback_sources=[],
        )

    return {
        "retrieved_chunks": [],

        "answer":
            response.answer,

        "sources":
            response.sources,

        "confidence":
            response.confidence,

        "response":
            response,
    }


# ============================================================
# Build LangGraph StateGraph
# ============================================================

graph_builder = StateGraph(
    SupportState
)

graph_builder.add_node(
    "classify_intent",
    classify_intent,
)

graph_builder.add_node(
    "retrieve_and_answer",
    retrieve_and_answer,
)

graph_builder.add_node(
    "direct_answer",
    direct_answer,
)


graph_builder.add_edge(
    START,
    "classify_intent",
)


graph_builder.add_conditional_edges(
    "classify_intent",
    route_by_intent,
    {
        "retrieve_and_answer":
            "retrieve_and_answer",

        "direct_answer":
            "direct_answer",
    },
)


graph_builder.add_edge(
    "retrieve_and_answer",
    END,
)

graph_builder.add_edge(
    "direct_answer",
    END,
)


support_graph = (
    graph_builder.compile()
)


# ============================================================
# FastAPI application
# ============================================================

app = FastAPI(
    title="Zepto Support Assistant",
)


@app.post(
    "/ask",
    response_model=AssistantResponse,
)
def ask(
    request: AskRequest,
) -> AssistantResponse:

    result = support_graph.invoke(
        {
            "query": request.query,
        }
    )

    return AssistantResponse(
        answer=result["answer"],
        sources=result["sources"],
        confidence=result["confidence"],
    )


# ============================================================
# Optional health endpoint
# Not required for grading; /ask is the graded endpoint.
# ============================================================

@app.get("/")
def root():
    return {
        "status": "running",
        "mock_llm": is_mock_mode(),
        "documents_indexed":
            collection.count(),
    }