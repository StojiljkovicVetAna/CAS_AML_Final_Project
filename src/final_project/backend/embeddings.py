"""Embedding providers for retrieval and ingestion."""

from __future__ import annotations

import hashlib
import logging
import math
from typing import Any, Protocol

import requests

from .config import BackendSettings


LOGGER = logging.getLogger(__name__)


def _setting(settings: BackendSettings, name: str, default: Any) -> Any:
    return getattr(settings, name, default)


class EmbeddingClient(Protocol):
    provider: str
    model: str

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        ...

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        ...

    def embed_text(self, text: str) -> list[float]:
        return self.embed_texts([text])[0]

    def embed_query(self, text: str) -> list[float]:
        return self.embed_text(text)


class BaseEmbeddingClient:
    def embed_text(self, text: str) -> list[float]:
        return self.embed_texts([text])[0]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self.embed_texts(texts)

    def embed_query(self, text: str) -> list[float]:
        return self.embed_text(text)


def _torch_dtype_from_name(dtype_name: str) -> Any:
    import torch

    normalized = dtype_name.strip().lower()
    dtype_map = {
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float16": torch.float16,
        "fp16": torch.float16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    if normalized not in dtype_map:
        raise ValueError(f"Unsupported EMBEDDING_TORCH_DTYPE: {dtype_name}")
    return dtype_map[normalized]


class GoogleEmbeddingClient(BaseEmbeddingClient):
    provider = "google"

    def __init__(self, settings: BackendSettings):
        self.model = settings.embedding_model
        self.api_key = settings.google_embedding_api_key or settings.google_api_key
        if not self.api_key:
            raise ValueError("GOOGLE_API_KEY_EMBED or GOOGLE_API_KEY is required")
        from google import genai
        from google.genai import types

        self._client = genai.Client(api_key=self.api_key)
        self._types = types

    def _embed_texts(self, texts: list[str], *, task_type: str | None = None) -> list[list[float]]:
        if not texts:
            return []
        config = None
        if task_type:
            config = self._types.EmbedContentConfig(task_type=task_type)
        response = self._client.models.embed_content(
            model=self.model,
            contents=texts,
            config=config,
        )
        return [embedding.values for embedding in response.embeddings]

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return self.embed_documents(texts)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._embed_texts(texts, task_type="RETRIEVAL_DOCUMENT")

    def embed_query(self, text: str) -> list[float]:
        return self._embed_texts([text], task_type="RETRIEVAL_QUERY")[0]


class OpenAICompatibleEmbeddingClient(BaseEmbeddingClient):
    provider = "openai_compatible"

    def __init__(self, settings: BackendSettings):
        self.model = settings.embedding_model
        self.base_url = settings.openai_embedding_base_url
        self.api_key = settings.openai_embedding_api_key
        if not self.api_key:
            raise ValueError("OPENAI_EMBEDDING_API_KEY is required")
        from openai import OpenAI

        kwargs = {"api_key": self.api_key}
        if self.base_url:
            kwargs["base_url"] = self.base_url
        self._client = OpenAI(**kwargs)

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        response = self._client.embeddings.create(model=self.model, input=texts)
        return [item.embedding for item in response.data]


class OllamaEmbeddingClient(BaseEmbeddingClient):
    provider = "ollama"

    def __init__(self, settings: BackendSettings):
        self.model = settings.embedding_model
        self.base_url = settings.ollama_base_url.rstrip("/")

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        response = requests.post(
            f"{self.base_url}/api/embed",
            json={"model": self.model, "input": texts},
            timeout=120,
        )
        response.raise_for_status()
        payload = response.json()
        embeddings = payload.get("embeddings")
        if not isinstance(embeddings, list):
            raise RuntimeError(f"Ollama response missing embeddings: {payload}")
        return embeddings


class SentenceTransformersEmbeddingClient(BaseEmbeddingClient):
    provider = "sentence_transformers"

    def __init__(self, settings: BackendSettings):
        self.model = settings.embedding_model
        self.batch_size_hint = None
        self.normalize_embeddings = _setting(settings, "embedding_normalize", True)
        self.show_progress = _setting(settings, "embedding_show_progress", True)
        if not self.model:
            raise ValueError("EMBEDDING_MODEL is required")
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover - optional local dependency
            raise RuntimeError(
                "sentence-transformers is required for EMBEDDING_PROVIDER=sentence_transformers"
            ) from exc

        kwargs = {"trust_remote_code": _setting(settings, "embedding_trust_remote_code", True)}
        embedding_device = _setting(settings, "embedding_device", "")
        if embedding_device:
            kwargs["device"] = embedding_device
        model_kwargs = {}
        embedding_torch_dtype = _setting(settings, "embedding_torch_dtype", "")
        if embedding_torch_dtype:
            model_kwargs["torch_dtype"] = _torch_dtype_from_name(embedding_torch_dtype)
        if model_kwargs:
            kwargs["model_kwargs"] = model_kwargs
        LOGGER.info("Loading SentenceTransformer model from %s.", self.model)
        self._model = SentenceTransformer(self.model, **kwargs)
        embedding_max_seq_length = _setting(settings, "embedding_max_seq_length", 0)
        if embedding_max_seq_length:
            self._model.max_seq_length = embedding_max_seq_length
        LOGGER.info(
            "Loaded SentenceTransformer model %s with max_seq_length=%s.",
            self.model,
            getattr(self._model, "max_seq_length", "unknown"),
        )

    def _encode(self, texts: list[str], *, mode: str) -> list[list[float]]:
        if not texts:
            return []
        encode_kwargs = {
            "batch_size": len(texts),
            "show_progress_bar": self.show_progress and len(texts) > 1,
            "normalize_embeddings": self.normalize_embeddings,
        }
        if mode == "query" and hasattr(self._model, "encode_query"):
            embeddings = self._model.encode_query(texts, **encode_kwargs)
        elif hasattr(self._model, "encode_document"):
            embeddings = self._model.encode_document(texts, **encode_kwargs)
        else:
            embeddings = self._model.encode(texts, convert_to_numpy=True, **encode_kwargs)
        return [embedding.tolist() for embedding in embeddings]

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return self.embed_documents(texts)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._encode(texts, mode="document")

    def embed_query(self, text: str) -> list[float]:
        return self._encode([text], mode="query")[0]


class HFTransformersEmbeddingClient(BaseEmbeddingClient):
    """Direct Hugging Face embedding path for text models with last-token pooling."""

    provider = "hf_transformers"

    def __init__(self, settings: BackendSettings):
        self.model = settings.embedding_model
        self.normalize_embeddings = _setting(settings, "embedding_normalize", True)
        self.max_seq_length = _setting(settings, "embedding_max_seq_length", 0) or None
        self.query_instruction = (
            "Instruct: Given a query, retrieve documents that answer the query \nQuery: "
            if "kalm" in self.model.lower()
            else ""
        )
        if not self.model:
            raise ValueError("EMBEDDING_MODEL is required")
        try:
            import torch
            from transformers import AutoModel, AutoTokenizer
        except ImportError as exc:  # pragma: no cover - optional local dependency
            raise RuntimeError(
                "torch and transformers are required for EMBEDDING_PROVIDER=hf_transformers"
            ) from exc

        self._torch = torch
        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model,
            trust_remote_code=_setting(settings, "embedding_trust_remote_code", True),
        )

        model_kwargs: dict[str, Any] = {
            "trust_remote_code": _setting(settings, "embedding_trust_remote_code", True)
        }
        embedding_torch_dtype = _setting(settings, "embedding_torch_dtype", "")
        if embedding_torch_dtype:
            model_kwargs["torch_dtype"] = _torch_dtype_from_name(embedding_torch_dtype)
        embedding_device = _setting(settings, "embedding_device", "")
        device_name = embedding_device.strip() if embedding_device else ""
        if device_name == "auto":
            model_kwargs["device_map"] = "auto"

        LOGGER.info("Loading Hugging Face transformer embedding model from %s.", self.model)
        self._model = AutoModel.from_pretrained(self.model, **model_kwargs)
        self._model.eval()
        if device_name and device_name != "auto":
            self._model.to(device_name)
        self._device = next(self._model.parameters()).device
        LOGGER.info(
            "Loaded Hugging Face transformer model %s on %s with max_seq_length=%s.",
            self.model,
            self._device,
            self.max_seq_length or "model default",
        )

    def _prepare_texts(self, texts: list[str], *, mode: str) -> list[str]:
        if mode == "query" and self.query_instruction:
            return [f"{self.query_instruction}{text}" for text in texts]
        return texts

    def _encode(self, texts: list[str], *, mode: str) -> list[list[float]]:
        if not texts:
            return []
        prepared_texts = self._prepare_texts(texts, mode=mode)
        tokenizer_kwargs: dict[str, Any] = {
            "padding": True,
            "truncation": True,
            "return_tensors": "pt",
        }
        if self.max_seq_length:
            tokenizer_kwargs["max_length"] = self.max_seq_length
        inputs = self._tokenizer(prepared_texts, **tokenizer_kwargs)
        inputs = {key: value.to(self._device) for key, value in inputs.items()}

        with self._torch.inference_mode():
            outputs = self._model(**inputs)
            hidden_states = outputs.last_hidden_state
            attention_mask = inputs["attention_mask"].to(hidden_states.device)
            last_token_indices = attention_mask.sum(dim=1) - 1
            batch_indices = self._torch.arange(hidden_states.shape[0], device=hidden_states.device)
            embeddings = hidden_states[batch_indices, last_token_indices]
            if self.normalize_embeddings:
                embeddings = self._torch.nn.functional.normalize(embeddings, p=2, dim=1)
        return embeddings.float().cpu().tolist()

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return self.embed_documents(texts)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._encode(texts, mode="document")

    def embed_query(self, text: str) -> list[float]:
        return self._encode([text], mode="query")[0]


class HashEmbeddingClient(BaseEmbeddingClient):
    """Deterministic tiny embedding client for tests and offline smoke checks."""

    provider = "hash"

    def __init__(self, settings: BackendSettings, dimensions: int = 64):
        self.model = settings.embedding_model or "hash"
        self.dimensions = dimensions

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        vectors = []
        for text in texts:
            vector = [0.0] * self.dimensions
            for token in text.lower().split():
                digest = hashlib.sha256(token.encode("utf-8")).digest()
                index = int.from_bytes(digest[:4], "big") % self.dimensions
                sign = 1.0 if digest[4] % 2 == 0 else -1.0
                vector[index] += sign
            norm = math.sqrt(sum(value * value for value in vector)) or 1.0
            vectors.append([value / norm for value in vector])
        return vectors


def get_embedding_client(settings: BackendSettings) -> EmbeddingClient:
    provider = settings.embedding_provider
    if provider == "google":
        return GoogleEmbeddingClient(settings)
    if provider in {"openai", "openai_compatible"}:
        return OpenAICompatibleEmbeddingClient(settings)
    if provider == "ollama":
        return OllamaEmbeddingClient(settings)
    if provider in {"sentence_transformers", "sentence-transformers", "local"}:
        return SentenceTransformersEmbeddingClient(settings)
    if provider in {"hf_transformers", "hf-transformers", "transformers"}:
        return HFTransformersEmbeddingClient(settings)
    if provider == "hash":
        return HashEmbeddingClient(settings)
    raise ValueError(f"Unsupported EMBEDDING_PROVIDER: {provider}")
