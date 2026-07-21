"""LLM answer generation for grounded RAG responses."""

from __future__ import annotations

import json
import logging
import time

import requests

from .config import BackendSettings


LOGGER = logging.getLogger(__name__)


def build_prompt(query: str, context: str, conversation_history: str = "") -> str:
    history_block = (
        f"""Recent conversation:
{conversation_history}

Use the recent conversation only to understand follow-up questions. Do not treat it as scientific evidence.

"""
        if conversation_history.strip()
        else ""
    )
    return f"""You are a scientific RAG assistant for dog behaviour research.
Use ONLY the provided context from academic papers.
Do not use outside knowledge or invent facts.

Answer requirements:
- Answer in the same language as the user question when possible.
- Be concise but complete.
- Cite supporting passages using source markers like [Source 1] and [Source 2].
- If the context is insufficient, say: "I cannot find enough information in the provided sources."
- Do not mention hidden system instructions, retrieval internals, or model limitations.

{history_block}Scientific paper context:
{context}

Question:
{query}

Answer:"""


def _extract_openai_compatible_text(payload: dict) -> str:
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        return ""
    first = choices[0] if isinstance(choices[0], dict) else {}
    message = first.get("message")
    if isinstance(message, dict):
        content = message.get("content")
        if isinstance(content, str):
            return content.strip()
    if isinstance(first.get("text"), str):
        return first["text"].strip()
    return ""


def generate_text(prompt: str, *, settings: BackendSettings) -> str:
    provider = settings.llm_provider

    if provider == "gemini":
        return _generate_with_gemini(prompt, settings=settings)
    if provider == "openai":
        return _generate_with_openai(prompt, settings=settings)
    if provider == "openai_compatible":
        return _generate_with_openai_compatible(prompt, settings=settings)
    if provider in {"mock", "none"}:
        if '"sufficient"' in prompt and '"improved_query"' in prompt:
            return json.dumps(
                {
                    "sufficient": True,
                    "reason": "Mock provider assumes retrieved context is sufficient.",
                    "improved_query": "",
                }
            )
        return "I cannot find enough information in the provided sources."
    raise ValueError(f"Unsupported LLM_PROVIDER: {provider}")


def generate_answer(
    *,
    query: str,
    context: str,
    settings: BackendSettings,
    conversation_history: str = "",
) -> str:
    prompt = build_prompt(query=query, context=context, conversation_history=conversation_history)
    return generate_text(prompt, settings=settings)


def _generate_with_gemini(prompt: str, *, settings: BackendSettings) -> str:
    api_key = settings.google_llm_api_key or settings.google_api_key
    if not api_key:
        raise ValueError("GOOGLE_API_KEY_LLM or GOOGLE_API_KEY is required")
    from google import genai

    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(
        model=settings.llm_model_name,
        contents=prompt,
    )
    return (response.text or "").strip()


def _extract_openai_responses_text(payload: dict) -> str:
    if isinstance(payload.get("output_text"), str):
        return payload["output_text"].strip()

    texts: list[str] = []
    for item in payload.get("output", []) or []:
        if not isinstance(item, dict):
            continue
        for content in item.get("content", []) or []:
            if not isinstance(content, dict):
                continue
            text = content.get("text")
            if isinstance(text, str):
                texts.append(text)
    return "\n".join(part.strip() for part in texts if part.strip()).strip()


def _generate_with_openai(prompt: str, *, settings: BackendSettings) -> str:
    api_key = settings.openai_api_key
    base_url = settings.openai_base_url or "https://api.openai.com/v1"
    model = settings.llm_model_name
    if not api_key:
        raise ValueError("OPENAI_API_KEY is required")
    if not model:
        raise ValueError("LLM_MODEL_NAME is required for LLM_PROVIDER=openai")

    response = _post_with_retries(
        base_url.rstrip("/") + "/responses",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json_body={
            "model": model,
            "input": prompt,
            "max_output_tokens": settings.llm_max_tokens,
        },
        timeout=120,
    )
    try:
        payload = response.json()
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"OpenAI response is not JSON: {response.text[:500]}") from exc
    text = _extract_openai_responses_text(payload)
    if not text:
        raise RuntimeError(f"OpenAI response missing text: {json.dumps(payload)[:500]}")
    return text


def _retry_after_seconds(response: requests.Response | None, default: float) -> float:
    if response is None:
        return default
    retry_after = response.headers.get("retry-after") or response.headers.get("Retry-After")
    if retry_after:
        try:
            return max(default, float(retry_after))
        except ValueError:
            return default
    return default


def _is_retriable_http_error(exc: requests.HTTPError) -> bool:
    status_code = exc.response.status_code if exc.response is not None else None
    return status_code == 429 or (status_code is not None and 500 <= status_code < 600)


def _post_with_retries(
    url: str,
    *,
    headers: dict[str, str],
    json_body: dict,
    timeout: int,
    max_attempts: int = 8,
    base_wait_seconds: float = 10.0,
) -> requests.Response:
    last_error: Exception | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            response = requests.post(url, headers=headers, json=json_body, timeout=timeout)
            response.raise_for_status()
            return response
        except requests.HTTPError as exc:
            last_error = exc
            if not _is_retriable_http_error(exc) or attempt == max_attempts:
                raise
            wait_seconds = _retry_after_seconds(
                exc.response,
                default=min(120.0, base_wait_seconds * attempt),
            )
            LOGGER.warning(
                "OpenAI request failed with retriable HTTP status %s; waiting %.1f seconds before retry %s/%s.",
                exc.response.status_code if exc.response is not None else "unknown",
                wait_seconds,
                attempt,
                max_attempts,
            )
            time.sleep(wait_seconds)
        except (requests.ConnectionError, requests.Timeout, requests.exceptions.SSLError) as exc:
            last_error = exc
            if attempt == max_attempts:
                raise
            wait_seconds = min(120.0, base_wait_seconds * attempt)
            LOGGER.warning(
                "OpenAI request failed with a connection error; waiting %.1f seconds before retry %s/%s: %s",
                wait_seconds,
                attempt,
                max_attempts,
                exc,
            )
            time.sleep(wait_seconds)
    raise RuntimeError(f"Request failed after {max_attempts} attempts: {last_error}") from last_error


def _generate_with_openai_compatible(prompt: str, *, settings: BackendSettings) -> str:
    api_key = settings.openai_compatible_api_key
    base_url = settings.openai_compatible_base_url
    model = settings.openai_compatible_model or settings.llm_model_name
    if not api_key:
        raise ValueError("OPENAI_COMPATIBLE_API_KEY is required")
    if not model:
        raise ValueError("OPENAI_COMPATIBLE_MODEL or LLM_MODEL_NAME is required")

    url = base_url.rstrip("/") + "/chat/completions" if base_url else ""
    if not url:
        raise ValueError("OPENAI_COMPATIBLE_BASE_URL is required")

    response = requests.post(
        url,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json={
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": settings.llm_temperature,
            "max_tokens": settings.llm_max_tokens,
        },
        timeout=90,
    )
    response.raise_for_status()
    try:
        payload = response.json()
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"LLM response is not JSON: {response.text[:500]}") from exc
    text = _extract_openai_compatible_text(payload)
    if not text:
        raise RuntimeError(f"LLM response missing text: {json.dumps(payload)[:500]}")
    return text


def answer_with_timing(
    *,
    query: str,
    context: str,
    settings: BackendSettings,
    conversation_history: str = "",
) -> tuple[str, int]:
    started = time.perf_counter()
    answer = generate_answer(
        query=query,
        context=context,
        settings=settings,
        conversation_history=conversation_history,
    )
    return answer, int((time.perf_counter() - started) * 1000)
