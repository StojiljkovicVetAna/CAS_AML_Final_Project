"""Final deployment retrieval profile.

This module keeps the selected production RAG strategy separate from the
generic retrieval/evaluation code. Experiments can still override these values,
but the deployed backend should use this profile unless there is a deliberate
methodological change.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class RagDeploymentProfile:
    name: str
    retrieve_top_k: int
    context_top_n: int
    reranker_provider: str
    allow_retrieval_overrides: bool
    description: str


FINAL_RAG_DEPLOYMENT_PROFILE = RagDeploymentProfile(
    name="reranked",
    retrieve_top_k=20,
    context_top_n=8,
    reranker_provider="jina",
    allow_retrieval_overrides=False,
    description=(
        "Final deployed pipeline: vector retrieval of 20 candidate chunks, "
        "Jina reranking, and the top 8 chunks passed to the LLM as context."
    ),
)


RERANKED_PIPELINES = {"reranked"}
DISABLED_RERANKER_PROVIDERS = {"", "none", "off", "false"}


def pipeline_uses_reranker(pipeline: str) -> bool:
    return pipeline.strip().lower() in RERANKED_PIPELINES
