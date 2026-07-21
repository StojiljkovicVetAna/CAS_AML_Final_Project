"""Environment-driven backend settings."""

from __future__ import annotations

from dataclasses import dataclass
import os

from .deployment import FINAL_RAG_DEPLOYMENT_PROFILE


def _get_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _get_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None or not value.strip():
        return default
    return int(value)


def _get_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None or not value.strip():
        return default
    return float(value)


@dataclass(frozen=True, slots=True)
class BackendSettings:
    """All runtime configuration for retrieval, reranking, and generation."""

    backend_api_key: str = ""
    allowed_origins: tuple[str, ...] = (
        "http://localhost:3000",
        "http://localhost:5173",
        "http://localhost:5174",
        "http://127.0.0.1:5173",
        "http://127.0.0.1:5174",
    )
    chat_debug_timings: bool = False
    rag_pipeline: str = FINAL_RAG_DEPLOYMENT_PROFILE.name
    allow_retrieval_overrides: bool = FINAL_RAG_DEPLOYMENT_PROFILE.allow_retrieval_overrides

    chroma_mode: str = "cloud"
    chroma_api_key: str = ""
    chroma_tenant: str = ""
    chroma_database: str = ""
    chroma_host: str = "api.trychroma.com"
    chroma_port: int = 8000
    chroma_ssl: bool = True
    chroma_collection: str = "dog_behavior_papers"
    chroma_persist_path: str = "data/chroma"

    embedding_provider: str = "google"
    embedding_model: str = "gemini-embedding-001"
    google_api_key: str = ""
    google_embedding_api_key: str = ""
    openai_embedding_base_url: str = ""
    openai_embedding_api_key: str = ""
    ollama_base_url: str = "http://localhost:11434"
    embedding_device: str = ""
    embedding_trust_remote_code: bool = True
    embedding_torch_dtype: str = ""
    embedding_max_seq_length: int = 0
    embedding_normalize: bool = True
    embedding_show_progress: bool = True

    retrieve_top_k: int = FINAL_RAG_DEPLOYMENT_PROFILE.retrieve_top_k
    context_top_n: int = FINAL_RAG_DEPLOYMENT_PROFILE.context_top_n

    reranker_provider: str = FINAL_RAG_DEPLOYMENT_PROFILE.reranker_provider
    jina_reranker_api_key: str = ""
    jina_reranker_url: str = "https://api.jina.ai/v1/rerank"
    jina_reranker_model: str = "jina-reranker-v3"

    llm_provider: str = "gemini"
    llm_model_name: str = "gemini-2.0-flash"
    google_llm_api_key: str = ""
    openai_api_key: str = ""
    openai_base_url: str = "https://api.openai.com/v1"
    openai_compatible_base_url: str = ""
    openai_compatible_api_key: str = ""
    openai_compatible_model: str = ""
    llm_temperature: float = 0.1
    llm_max_tokens: int = 900


def load_settings() -> BackendSettings:
    origins = tuple(
        origin.strip()
        for origin in os.getenv(
            "ALLOWED_ORIGINS",
            (
                "http://localhost:3000,http://localhost:5173,http://localhost:5174,"
                "http://127.0.0.1:5173,http://127.0.0.1:5174"
            ),
        ).split(",")
        if origin.strip()
    )
    return BackendSettings(
        backend_api_key=os.getenv("BACKEND_API_KEY", ""),
        allowed_origins=origins,
        chat_debug_timings=_get_bool("CHAT_DEBUG_TIMINGS"),
        rag_pipeline=os.getenv("RAG_PIPELINE", FINAL_RAG_DEPLOYMENT_PROFILE.name)
        .strip()
        .lower(),
        allow_retrieval_overrides=_get_bool(
            "ALLOW_RETRIEVAL_OVERRIDES",
            FINAL_RAG_DEPLOYMENT_PROFILE.allow_retrieval_overrides,
        ),
        chroma_mode=os.getenv("CHROMA_MODE", "cloud").strip().lower(),
        chroma_api_key=os.getenv("CHROMA_API_KEY", ""),
        chroma_tenant=os.getenv("CHROMA_TENANT", ""),
        chroma_database=os.getenv("CHROMA_DATABASE", ""),
        chroma_host=os.getenv("CHROMA_HOST", "api.trychroma.com"),
        chroma_port=_get_int("CHROMA_PORT", 8000),
        chroma_ssl=_get_bool("CHROMA_SSL", True),
        chroma_collection=os.getenv("CHROMA_COLLECTION", "dog_behavior_papers"),
        chroma_persist_path=os.getenv("CHROMA_PERSIST_PATH", "data/chroma"),
        embedding_provider=os.getenv("EMBEDDING_PROVIDER", "google").strip().lower(),
        embedding_model=os.getenv("EMBEDDING_MODEL", "gemini-embedding-001"),
        google_api_key=os.getenv("GOOGLE_API_KEY", ""),
        google_embedding_api_key=os.getenv("GOOGLE_API_KEY_EMBED", ""),
        openai_embedding_base_url=os.getenv("OPENAI_EMBEDDING_BASE_URL", ""),
        openai_embedding_api_key=os.getenv("OPENAI_EMBEDDING_API_KEY", ""),
        ollama_base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        embedding_device=os.getenv("EMBEDDING_DEVICE", ""),
        embedding_trust_remote_code=_get_bool("EMBEDDING_TRUST_REMOTE_CODE", True),
        embedding_torch_dtype=os.getenv("EMBEDDING_TORCH_DTYPE", ""),
        embedding_max_seq_length=_get_int("EMBEDDING_MAX_SEQ_LENGTH", 0),
        embedding_normalize=_get_bool("EMBEDDING_NORMALIZE", True),
        embedding_show_progress=_get_bool("EMBEDDING_SHOW_PROGRESS", True),
        retrieve_top_k=_get_int("RETRIEVE_TOP_K", FINAL_RAG_DEPLOYMENT_PROFILE.retrieve_top_k),
        context_top_n=_get_int("CONTEXT_TOP_N", FINAL_RAG_DEPLOYMENT_PROFILE.context_top_n),
        reranker_provider=os.getenv(
            "RERANKER_PROVIDER",
            FINAL_RAG_DEPLOYMENT_PROFILE.reranker_provider,
        )
        .strip()
        .lower(),
        jina_reranker_api_key=os.getenv("JINA_RERANKER_API_KEY", ""),
        jina_reranker_url=os.getenv("JINA_RERANKER_URL", "https://api.jina.ai/v1/rerank"),
        jina_reranker_model=os.getenv("JINA_RERANKER_MODEL", "jina-reranker-v3"),
        llm_provider=os.getenv("LLM_PROVIDER", "gemini").strip().lower(),
        llm_model_name=os.getenv("LLM_MODEL_NAME", "gemini-2.0-flash"),
        google_llm_api_key=os.getenv("GOOGLE_API_KEY_LLM", ""),
        openai_api_key=os.getenv("OPENAI_API_KEY", ""),
        openai_base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
        openai_compatible_base_url=os.getenv("OPENAI_COMPATIBLE_BASE_URL", ""),
        openai_compatible_api_key=os.getenv("OPENAI_COMPATIBLE_API_KEY", ""),
        openai_compatible_model=os.getenv("OPENAI_COMPATIBLE_MODEL", ""),
        llm_temperature=_get_float("LLM_TEMPERATURE", 0.1),
        llm_max_tokens=_get_int("LLM_MAX_TOKENS", 900),
    )


def active_llm_model(settings: BackendSettings) -> str:
    """Return the model name used by the configured LLM provider."""
    if settings.llm_provider == "openai_compatible":
        return settings.openai_compatible_model or settings.llm_model_name
    return settings.llm_model_name
