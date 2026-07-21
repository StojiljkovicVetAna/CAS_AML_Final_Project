"""FastAPI application for the final RAG backend."""

from __future__ import annotations

import time
from typing import Any

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from .config import BackendSettings, active_llm_model, load_settings
from .deployment import DISABLED_RERANKER_PROVIDERS, pipeline_uses_reranker
from .embeddings import get_embedding_client
from .llm import answer_with_timing
from .retrieval import format_context_for_llm, retrieve_chunks
from .schemas import (
    ChatRequest,
    ChatResponse,
    ConversationTurn,
    HealthResponse,
    RetrievedChunk,
    SearchRequest,
    SearchResponse,
    Source,
)


load_dotenv()
settings = load_settings()

app = FastAPI(title="Dog Behaviour RAG Backend", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=list(settings.allowed_origins),
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
    max_age=600,
)


def require_api_key(x_api_key: str | None = Header(default=None)) -> None:
    if not settings.backend_api_key:
        return
    if x_api_key != settings.backend_api_key:
        raise HTTPException(status_code=401, detail="Valid X-API-Key header required")


def _source_from_chunk(chunk: dict[str, Any]) -> Source:
    return Source(
        chunk_id=chunk.get("chunk_id", ""),
        document_id=chunk.get("document_id", ""),
        title=chunk.get("title", ""),
        authors=chunk.get("authors", []),
        doi=chunk.get("doi", ""),
        journal=chunk.get("journal", ""),
        publication_date=chunk.get("publication_date", ""),
        section_type=chunk.get("section_type", ""),
        heading=chunk.get("heading", ""),
        chunk_index=chunk.get("chunk_index", 0),
        distance=chunk.get("distance"),
        rerank_score=chunk.get("rerank_score"),
    )


def _retrieved_from_chunk(chunk: dict[str, Any]) -> RetrievedChunk:
    return RetrievedChunk(text=chunk.get("text", ""), **_source_from_chunk(chunk).model_dump())


def _retrieval_overrides(
    request: ChatRequest | SearchRequest,
) -> tuple[int | None, int | None, bool | None]:
    if not settings.allow_retrieval_overrides:
        return None, None, None
    return request.top_k, request.top_n, request.use_reranker


def _active_reranker_provider(use_reranker: bool | None) -> str:
    enabled = (
        pipeline_uses_reranker(settings.rag_pipeline)
        and settings.reranker_provider not in DISABLED_RERANKER_PROVIDERS
        if use_reranker is None
        else use_reranker
    )
    return settings.reranker_provider if enabled else "none"


def _format_conversation_history(turns: list[ConversationTurn]) -> str:
    if not turns:
        return ""
    recent_turns = turns[-6:]
    lines = []
    for turn in recent_turns:
        content = " ".join(turn.content.strip().split())
        if len(content) > 1200:
            content = content[:1200].rsplit(" ", 1)[0] + " ..."
        label = "User" if turn.role == "user" else "Assistant"
        lines.append(f"{label}: {content}")
    return "\n".join(lines)


def _history_aware_retrieval_query(query: str, turns: list[ConversationTurn]) -> str:
    user_turns = [
        " ".join(turn.content.strip().split())
        for turn in turns[-6:]
        if turn.role == "user" and turn.content.strip()
    ]
    if not user_turns:
        return query
    recent_questions = "\n".join(f"- {turn[:500]}" for turn in user_turns[-3:])
    return f"""Recent user questions:
{recent_questions}

Current question:
{query}
"""


@app.api_route("/", methods=["GET", "HEAD"])
def root() -> dict[str, str]:
    return {"status": "ok", "service": "dog-behaviour-rag-backend"}


@app.get("/api/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        service="dog-behaviour-rag-backend",
        config={
            "chroma_mode": settings.chroma_mode,
            "chroma_collection": settings.chroma_collection,
            "embedding_provider": settings.embedding_provider,
            "embedding_model": settings.embedding_model,
            "rag_pipeline": settings.rag_pipeline,
            "retrieve_top_k": settings.retrieve_top_k,
            "context_top_n": settings.context_top_n,
            "allow_retrieval_overrides": settings.allow_retrieval_overrides,
            "reranker_provider": settings.reranker_provider,
            "llm_provider": settings.llm_provider,
            "llm_model": active_llm_model(settings),
        },
    )


@app.post(
    "/api/search",
    response_model=SearchResponse,
    dependencies=[Depends(require_api_key)],
)
def search(request: SearchRequest) -> SearchResponse:
    query = request.query.strip()
    if not query:
        raise HTTPException(status_code=400, detail="Query is empty")

    try:
        top_k, top_n, use_reranker = _retrieval_overrides(request)
        retrieval_query = _history_aware_retrieval_query(query, request.conversation_history)
        chunks, timings_ms = retrieve_chunks(
            retrieval_query,
            settings=settings,
            top_k=top_k,
            top_n=top_n,
            use_reranker=use_reranker,
        )
        embedding_client = get_embedding_client(settings)
        return SearchResponse(
            chunks=[_retrieved_from_chunk(chunk) for chunk in chunks],
            embedding_provider=embedding_client.provider,
            embedding_model=embedding_client.model,
            reranker_provider=_active_reranker_provider(use_reranker),
            timings_ms=timings_ms if settings.chat_debug_timings else None,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post(
    "/api/chat",
    response_model=ChatResponse,
    dependencies=[Depends(require_api_key)],
)
def chat(request: ChatRequest) -> ChatResponse:
    query = request.query.strip()
    if not query:
        raise HTTPException(status_code=400, detail="Query is empty")

    timings_ms: dict[str, int] = {}
    started = time.perf_counter()
    try:
        top_k, top_n, use_reranker = _retrieval_overrides(request)
        conversation_history = _format_conversation_history(request.conversation_history)
        retrieval_query = _history_aware_retrieval_query(query, request.conversation_history)
        chunks, retrieval_timings = retrieve_chunks(
            retrieval_query,
            settings=settings,
            top_k=top_k,
            top_n=top_n,
            use_reranker=use_reranker,
        )
        timings_ms.update(retrieval_timings)

        if not chunks:
            answer = "I cannot find enough information in the provided sources."
            generation_ms = 0
        else:
            context = format_context_for_llm(chunks)
            answer, generation_ms = answer_with_timing(
                query=query,
                context=context,
                settings=settings,
                conversation_history=conversation_history,
            )
        timings_ms["generate_answer"] = generation_ms
        timings_ms["total"] = int((time.perf_counter() - started) * 1000)

        embedding_client = get_embedding_client(settings)
        return ChatResponse(
            answer=answer,
            sources=[_source_from_chunk(chunk) for chunk in chunks],
            llm_provider=settings.llm_provider,
            llm_model=active_llm_model(settings),
            embedding_provider=embedding_client.provider,
            embedding_model=embedding_client.model,
            reranker_provider=_active_reranker_provider(use_reranker),
            timings_ms=timings_ms if settings.chat_debug_timings else None,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
