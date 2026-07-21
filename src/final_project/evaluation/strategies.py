"""Comparable RAG retrieval strategies for final evaluation."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import re
import time
from typing import Any

from final_project.backend.config import BackendSettings
from final_project.backend.llm import answer_with_timing, generate_text
from final_project.backend.retrieval import format_context_for_llm, retrieve_chunks


@dataclass(slots=True)
class StrategyResult:
    strategy: str
    question_id: str
    question: str
    answer: str
    chunks: list[dict[str, Any]]
    retrieval_query: str
    retrieval_queries: list[str] = field(default_factory=list)
    rewritten_query: str = ""
    context_sufficient: bool | None = None
    second_retrieval_used: bool = False
    timings_ms: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "strategy": self.strategy,
            "question_id": self.question_id,
            "question": self.question,
            "answer": self.answer,
            "retrieval_query": self.retrieval_query,
            "rewritten_query": self.rewritten_query,
            "context_sufficient": self.context_sufficient,
            "second_retrieval_used": self.second_retrieval_used,
            "chunks": self.chunks,
            "retrieval_queries": self.retrieval_queries or [self.retrieval_query],
            "timings_ms": self.timings_ms,
        }


def run_classic_rag(
    *,
    question_id: str,
    question: str,
    settings: BackendSettings,
    context_k: int,
) -> StrategyResult:
    started = time.perf_counter()
    chunks, timings = retrieve_chunks(
        question,
        settings=settings,
        top_k=context_k,
        top_n=context_k,
        use_reranker=False,
    )
    answer, answer_ms = answer_with_timing(
        query=question,
        context=format_context_for_llm(chunks),
        settings=settings,
    )
    timings["generate_answer"] = answer_ms
    timings["total"] = int((time.perf_counter() - started) * 1000)
    return StrategyResult(
        strategy="classic",
        question_id=question_id,
        question=question,
        answer=answer,
        chunks=chunks,
        retrieval_query=question,
        retrieval_queries=[question],
        timings_ms=timings,
    )


def run_reranked_rag(
    *,
    question_id: str,
    question: str,
    settings: BackendSettings,
    candidate_k: int,
    context_k: int,
) -> StrategyResult:
    started = time.perf_counter()
    chunks, timings = retrieve_chunks(
        question,
        settings=settings,
        top_k=candidate_k,
        top_n=context_k,
        use_reranker=True,
    )
    answer, answer_ms = answer_with_timing(
        query=question,
        context=format_context_for_llm(chunks),
        settings=settings,
    )
    timings["generate_answer"] = answer_ms
    timings["total"] = int((time.perf_counter() - started) * 1000)
    return StrategyResult(
        strategy="reranked",
        question_id=question_id,
        question=question,
        answer=answer,
        chunks=chunks,
        retrieval_query=question,
        retrieval_queries=[question],
        timings_ms=timings,
    )


def _hyde_prompt(question: str, sentence_count: int) -> str:
    return f"""You are the vector database retrieval helper of a scientific RAG system.
Your task is to take a user question and write one hypothetical paragraph from an academic paper that would answer that question.
The paragraph should have about {sentence_count} sentences and will be embedded to retrieve real relevant chunks from the vector database.

Rules:
- Write only the hypothetical academic paragraph.
- Use scientific terminology likely to appear in dog behaviour research papers.
- Do not cite sources or invent author names.
- Preserve the meaning of the original question.

Question:
{question}
"""


def _multi_query_prompt(question: str, num_queries: int) -> str:
    return f"""You are planning retrieval for a scientific RAG system about dog behaviour research.
Generate {num_queries} complementary retrieval queries for the user question.

Return ONLY valid JSON with this schema:
{{
  "queries": [
    "query 1",
    "query 2",
    "query 3"
  ]
}}

Rules:
- Include the original intent.
- Use concise academic search language.
- Make the queries complementary: one direct query, one scientific keyword query, and one synonym/entity-expanded query.
- Do not answer the question.

Question:
{question}
"""


def _parse_query_list(text: str, fallback_query: str, max_queries: int) -> list[str]:
    cleaned = text.strip()
    if "```" in cleaned:
        cleaned = re.sub(r"^```(?:json)?", "", cleaned).strip()
        cleaned = re.sub(r"```$", "", cleaned).strip()

    queries: list[str] = []
    try:
        payload = json.loads(cleaned)
    except json.JSONDecodeError:
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start >= 0 and end > start:
            try:
                payload = json.loads(cleaned[start : end + 1])
            except json.JSONDecodeError:
                payload = {}
        else:
            payload = {}

    if isinstance(payload, dict) and isinstance(payload.get("queries"), list):
        queries = [str(item).strip() for item in payload["queries"] if str(item).strip()]
    elif isinstance(payload, list):
        queries = [str(item).strip() for item in payload if str(item).strip()]

    if not queries:
        queries = [fallback_query]
    if fallback_query not in queries:
        queries.insert(0, fallback_query)

    deduped: list[str] = []
    seen: set[str] = set()
    for query in queries:
        key = re.sub(r"\s+", " ", query.lower()).strip()
        if key and key not in seen:
            seen.add(key)
            deduped.append(query)
    return deduped[:max_queries]


def _fuse_ranked_chunks(ranked_lists: list[list[dict[str, Any]]], *, top_n: int) -> list[dict[str, Any]]:
    by_id: dict[str, dict[str, Any]] = {}
    scores: dict[str, float] = {}
    first_seen: dict[str, int] = {}
    sequence = 0
    k = 60

    for ranked in ranked_lists:
        for rank, chunk in enumerate(ranked, start=1):
            chunk_key = chunk.get("chunk_id") or chunk.get("text", "")
            if not chunk_key:
                continue
            if chunk_key not in by_id:
                by_id[chunk_key] = dict(chunk)
                first_seen[chunk_key] = sequence
                sequence += 1
            scores[chunk_key] = scores.get(chunk_key, 0.0) + 1.0 / (k + rank)

    ordered = sorted(scores, key=lambda key: (-scores[key], first_seen[key]))
    fused = []
    for key in ordered[:top_n]:
        chunk = dict(by_id[key])
        chunk["fusion_score"] = scores[key]
        fused.append(chunk)
    return fused


def run_hyde_rag(
    *,
    question_id: str,
    question: str,
    settings: BackendSettings,
    context_k: int,
    hyde_sentence_count: int,
) -> StrategyResult:
    started = time.perf_counter()
    timings: dict[str, int] = {}

    t0 = time.perf_counter()
    hypothetical_document = generate_text(
        _hyde_prompt(question, hyde_sentence_count),
        settings=settings,
    )
    timings["generate_hypothetical_document"] = int((time.perf_counter() - t0) * 1000)

    chunks, retrieval_timings = retrieve_chunks(
        hypothetical_document,
        settings=settings,
        top_k=context_k,
        top_n=context_k,
        use_reranker=False,
    )
    timings.update(retrieval_timings)
    answer, answer_ms = answer_with_timing(
        query=question,
        context=format_context_for_llm(chunks),
        settings=settings,
    )
    timings["generate_answer"] = answer_ms
    timings["total"] = int((time.perf_counter() - started) * 1000)
    return StrategyResult(
        strategy="hyde",
        question_id=question_id,
        question=question,
        answer=answer,
        chunks=chunks,
        retrieval_query=hypothetical_document,
        retrieval_queries=[hypothetical_document],
        rewritten_query=hypothetical_document,
        timings_ms=timings,
    )


def run_hyde_reranked_rag(
    *,
    question_id: str,
    question: str,
    settings: BackendSettings,
    candidate_k: int,
    context_k: int,
    hyde_sentence_count: int,
) -> StrategyResult:
    started = time.perf_counter()
    timings: dict[str, int] = {}

    t0 = time.perf_counter()
    hypothetical_document = generate_text(
        _hyde_prompt(question, hyde_sentence_count),
        settings=settings,
    )
    timings["generate_hypothetical_document"] = int((time.perf_counter() - t0) * 1000)

    chunks, retrieval_timings = retrieve_chunks(
        hypothetical_document,
        settings=settings,
        top_k=candidate_k,
        top_n=context_k,
        use_reranker=True,
    )
    timings.update(retrieval_timings)
    answer, answer_ms = answer_with_timing(
        query=question,
        context=format_context_for_llm(chunks),
        settings=settings,
    )
    timings["generate_answer"] = answer_ms
    timings["total"] = int((time.perf_counter() - started) * 1000)
    return StrategyResult(
        strategy="hyde_reranked",
        question_id=question_id,
        question=question,
        answer=answer,
        chunks=chunks,
        retrieval_query=hypothetical_document,
        retrieval_queries=[hypothetical_document],
        rewritten_query=hypothetical_document,
        timings_ms=timings,
    )


def run_multi_query_rag(
    *,
    question_id: str,
    question: str,
    settings: BackendSettings,
    context_k: int,
    num_queries: int,
    per_query_k: int,
) -> StrategyResult:
    started = time.perf_counter()
    timings: dict[str, int] = {}

    t0 = time.perf_counter()
    query_text = generate_text(
        _multi_query_prompt(question, num_queries),
        settings=settings,
    )
    queries = _parse_query_list(query_text, fallback_query=question, max_queries=num_queries)
    timings["generate_queries"] = int((time.perf_counter() - t0) * 1000)

    ranked_lists: list[list[dict[str, Any]]] = []
    for index, query in enumerate(queries, start=1):
        chunks, retrieval_timings = retrieve_chunks(
            query,
            settings=settings,
            top_k=per_query_k,
            top_n=per_query_k,
            use_reranker=False,
        )
        ranked_lists.append(chunks)
        for key, value in retrieval_timings.items():
            timings[f"query_{index}_{key}"] = value

    chunks = _fuse_ranked_chunks(ranked_lists, top_n=context_k)
    answer, answer_ms = answer_with_timing(
        query=question,
        context=format_context_for_llm(chunks),
        settings=settings,
    )
    timings["generate_answer"] = answer_ms
    timings["total"] = int((time.perf_counter() - started) * 1000)
    return StrategyResult(
        strategy="multi_query",
        question_id=question_id,
        question=question,
        answer=answer,
        chunks=chunks,
        retrieval_query=queries[0],
        retrieval_queries=queries,
        rewritten_query=json.dumps(queries, ensure_ascii=False),
        timings_ms=timings,
    )


def _agent_prompt(question: str, context: str) -> str:
    return f"""You are controlling a retrieval step for a scientific RAG system.
Assess whether the context is sufficient to answer the question.

Return ONLY valid JSON with this schema:
{{
  "sufficient": true or false,
  "reason": "short reason",
  "improved_query": "a better search query if insufficient, otherwise the original question"
}}

Rules:
- Say sufficient=true only if the context contains enough evidence to answer reliably.
- If insufficient, write a concise improved retrieval query with important terms and synonyms.
- Do not answer the user question.

Question:
{question}

Context:
{context}
"""


def _parse_agent_decision(text: str, fallback_query: str) -> dict[str, Any]:
    cleaned = text.strip()
    if "```" in cleaned:
        cleaned = re.sub(r"^```(?:json)?", "", cleaned).strip()
        cleaned = re.sub(r"```$", "", cleaned).strip()
    try:
        payload = json.loads(cleaned)
    except json.JSONDecodeError:
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start >= 0 and end > start:
            try:
                payload = json.loads(cleaned[start : end + 1])
            except json.JSONDecodeError:
                payload = {}
        else:
            payload = {}
    return {
        "sufficient": bool(payload.get("sufficient", False)),
        "reason": str(payload.get("reason", "")),
        "improved_query": str(payload.get("improved_query") or fallback_query),
    }


def run_agentic_rag(
    *,
    question_id: str,
    question: str,
    settings: BackendSettings,
    initial_k: int,
    context_k: int,
    second_candidate_k: int,
    rerank_second_retrieval: bool,
) -> StrategyResult:
    started = time.perf_counter()
    chunks, timings = retrieve_chunks(
        question,
        settings=settings,
        top_k=initial_k,
        top_n=context_k,
        use_reranker=False,
    )

    t0 = time.perf_counter()
    decision_text = generate_text(
        _agent_prompt(question, format_context_for_llm(chunks)),
        settings=settings,
    )
    decision = _parse_agent_decision(decision_text, fallback_query=question)
    timings["judge_context"] = int((time.perf_counter() - t0) * 1000)

    retrieval_query = question
    rewritten_query = ""
    second_retrieval_used = False
    if not decision["sufficient"]:
        rewritten_query = decision["improved_query"]
        retrieval_query = rewritten_query
        second_retrieval_used = True
        second_chunks, second_timings = retrieve_chunks(
            rewritten_query,
            settings=settings,
            top_k=second_candidate_k,
            top_n=context_k,
            use_reranker=rerank_second_retrieval,
        )
        chunks = second_chunks
        for key, value in second_timings.items():
            timings[f"second_{key}"] = value

    answer, answer_ms = answer_with_timing(
        query=question,
        context=format_context_for_llm(chunks),
        settings=settings,
    )
    timings["generate_answer"] = answer_ms
    timings["total"] = int((time.perf_counter() - started) * 1000)
    return StrategyResult(
        strategy="agentic",
        question_id=question_id,
        question=question,
        answer=answer,
        chunks=chunks,
        retrieval_query=retrieval_query,
        retrieval_queries=[retrieval_query],
        rewritten_query=rewritten_query,
        context_sufficient=decision["sufficient"],
        second_retrieval_used=second_retrieval_used,
        timings_ms=timings,
    )
