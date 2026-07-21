"""Pydantic API schemas for the RAG backend."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class ConversationTurn(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(..., min_length=1, max_length=4000)


class ChatRequest(BaseModel):
    query: str = Field(..., min_length=1)
    conversation_history: list[ConversationTurn] = Field(default_factory=list, max_length=6)
    top_k: int | None = Field(default=None, ge=1, le=100)
    top_n: int | None = Field(default=None, ge=1, le=30)
    use_reranker: bool | None = None


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1)
    conversation_history: list[ConversationTurn] = Field(default_factory=list, max_length=6)
    top_k: int | None = Field(default=None, ge=1, le=100)
    top_n: int | None = Field(default=None, ge=1, le=30)
    use_reranker: bool | None = None


class Source(BaseModel):
    chunk_id: str
    document_id: str
    title: str = ""
    authors: list[str] = Field(default_factory=list)
    doi: str = ""
    journal: str = ""
    publication_date: str = ""
    section_type: str = ""
    heading: str = ""
    chunk_index: int = 0
    distance: float | None = None
    rerank_score: float | None = None


class RetrievedChunk(Source):
    text: str


class ChatResponse(BaseModel):
    answer: str
    sources: list[Source]
    llm_provider: str
    llm_model: str
    embedding_provider: str
    embedding_model: str
    reranker_provider: str
    timings_ms: dict[str, int] | None = None


class SearchResponse(BaseModel):
    chunks: list[RetrievedChunk]
    embedding_provider: str
    embedding_model: str
    reranker_provider: str
    timings_ms: dict[str, int] | None = None


class HealthResponse(BaseModel):
    status: str
    service: str
    config: dict[str, Any]
