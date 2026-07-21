import json

from final_project.backend.config import BackendSettings, load_settings
from final_project.backend.deployment import FINAL_RAG_DEPLOYMENT_PROFILE
from final_project.backend.embeddings import HashEmbeddingClient
from final_project.backend.llm import build_prompt
from final_project.backend.retrieval import format_context_for_llm, retrieve_chunks
from final_project.backend.rerankers import NoReranker
from final_project.backend.upload import check_upload_ready, chunk_metadata, load_chunks, upload_chunks


class FakeCollection:
    def __init__(self):
        self.upserts = []
        self.existing_ids = set()

    def upsert(self, **kwargs):
        self.upserts.append(kwargs)

    def get(self, ids=None, include=None):
        ids = ids or []
        return {"ids": [item for item in ids if item in self.existing_ids]}


class FakeQueryCollection:
    def __init__(self):
        self.query_embeddings = []

    def query(self, **kwargs):
        self.query_embeddings.append(kwargs["query_embeddings"][0])
        return {
            "documents": [["Retrieved sentence."]],
            "metadatas": [[{"chunk_id": "chunk-1", "title": "Paper"}]],
            "distances": [[0.12]],
        }


class FakeEmbeddingClient:
    provider = "fake"
    model = "fake-model"

    def __init__(self):
        self.document_calls = []
        self.query_calls = []

    def embed_texts(self, texts):
        return self.embed_documents(texts)

    def embed_documents(self, texts):
        self.document_calls.append(texts)
        return [[1.0, 0.0] for _ in texts]

    def embed_text(self, text):
        return self.embed_query(text)

    def embed_query(self, text):
        self.query_calls.append(text)
        return [0.0, 1.0]


def test_chunk_metadata_is_chroma_safe():
    chunk = {
        "chunk_id": "chunk-1",
        "document_id": "paper-1",
        "chunk_index": 3,
        "section_type": "body",
        "heading": "Results",
        "text": "Dogs looked longer.",
        "metadata": {
            "title": "A paper",
            "authors": ["A. Smith", "B. Jones"],
            "doi": "10.123/example",
        },
        "provenance": {"chunker": "section_aware_academic_v1"},
    }

    metadata = chunk_metadata(chunk)

    assert metadata["chunk_id"] == "chunk-1"
    assert metadata["title"] == "A paper"
    assert metadata["authors"] == json.dumps(["A. Smith", "B. Jones"])
    assert all(isinstance(value, (str, int, float, bool)) for value in metadata.values())


def test_upload_chunks_uses_embeddings_and_upsert(tmp_path):
    chunks_path = tmp_path / "chunks.jsonl"
    chunks_path.write_text(
        json.dumps(
            {
                "chunk_id": "chunk-1",
                "document_id": "paper-1",
                "text": "Dogs can follow human pointing gestures.",
                "chunk_index": 0,
                "section_type": "abstract",
                "metadata": {"title": "Pointing paper"},
                "provenance": {},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    settings = BackendSettings(
        embedding_provider="hash",
        embedding_model="hash",
        chroma_collection="test_collection",
    )
    collection = FakeCollection()
    embedding_client = FakeEmbeddingClient()

    manifest = upload_chunks(
        chunks_path,
        settings=settings,
        embedding_client=embedding_client,
        collection=collection,
    )

    assert manifest["num_uploaded"] == 1
    assert embedding_client.document_calls == [["Dogs can follow human pointing gestures."]]
    assert collection.upserts[0]["ids"] == ["chunk-1"]
    assert collection.upserts[0]["documents"] == ["Dogs can follow human pointing gestures."]
    assert collection.upserts[0]["embeddings"] == [[1.0, 0.0]]


def test_upload_chunks_can_skip_existing_ids(tmp_path):
    chunks_path = tmp_path / "chunks.jsonl"
    rows = [
        {"chunk_id": "chunk-1", "document_id": "paper-1", "text": "Already uploaded."},
        {"chunk_id": "chunk-2", "document_id": "paper-1", "text": "Needs upload."},
    ]
    chunks_path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    settings = BackendSettings(
        embedding_provider="hash",
        embedding_model="hash",
        chroma_collection="test_collection",
    )
    collection = FakeCollection()
    collection.existing_ids = {"chunk-1"}
    embedding_client = FakeEmbeddingClient()

    manifest = upload_chunks(
        chunks_path,
        settings=settings,
        embedding_client=embedding_client,
        collection=collection,
        skip_existing=True,
    )

    assert embedding_client.document_calls == [["Needs upload."]]
    assert collection.upserts[0]["ids"] == ["chunk-2"]
    assert manifest["num_uploaded"] == 1
    assert manifest["num_skipped_existing"] == 1
    assert manifest["skip_existing"] is True


def test_check_upload_ready_embeds_document_and_query(tmp_path):
    chunks_path = tmp_path / "chunks.jsonl"
    chunks_path.write_text(
        json.dumps(
            {
                "chunk_id": "chunk-1",
                "document_id": "paper-1",
                "text": "Dogs can follow human pointing gestures.",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    settings = BackendSettings(
        embedding_provider="google",
        embedding_model="gemini-embedding-001",
        chroma_collection="dog_behavior_papers_gemini",
    )
    embedding_client = FakeEmbeddingClient()
    collection = FakeCollection()
    collection.count = lambda: 12

    manifest = check_upload_ready(
        chunks_path,
        settings=settings,
        embedding_client=embedding_client,
        collection=collection,
    )

    assert embedding_client.document_calls == [["Dogs can follow human pointing gestures."]]
    assert embedding_client.query_calls == ["What does the literature say about dog behaviour?"]
    assert manifest["sample_document_embedding_dimensions"] == 2
    assert manifest["sample_query_embedding_dimensions"] == 2
    assert manifest["collection"] == "dog_behavior_papers_gemini"
    assert manifest["collection_count"] == 12


def test_hash_embedding_client_supports_single_text_embedding():
    settings = BackendSettings(embedding_provider="hash", embedding_model="hash")
    client = HashEmbeddingClient(settings)

    assert client.embed_text("dog") == client.embed_texts(["dog"])[0]
    assert client.embed_query("dog") == client.embed_text("dog")


def test_retrieve_chunks_uses_query_embedding():
    settings = BackendSettings(
        embedding_provider="hash",
        embedding_model="hash",
        reranker_provider="none",
        retrieve_top_k=1,
        context_top_n=1,
    )
    embedding_client = FakeEmbeddingClient()
    collection = FakeQueryCollection()

    chunks, _timings = retrieve_chunks(
        "What do dogs understand?",
        settings=settings,
        embedding_client=embedding_client,
        collection=collection,
    )

    assert embedding_client.query_calls == ["What do dogs understand?"]
    assert embedding_client.document_calls == []
    assert collection.query_embeddings == [[0.0, 1.0]]
    assert chunks[0]["text"] == "Retrieved sentence."


def test_load_chunks_skips_empty_text(tmp_path):
    chunks_path = tmp_path / "chunks.jsonl"
    chunks_path.write_text(
        json.dumps({"chunk_id": "keep", "text": "text"}) + "\n"
        + json.dumps({"chunk_id": "skip", "text": ""}) + "\n",
        encoding="utf-8",
    )

    chunks = load_chunks(chunks_path)

    assert [chunk["chunk_id"] for chunk in chunks] == ["keep"]


def test_no_reranker_keeps_vector_order():
    chunks = [{"text": "a"}, {"text": "b"}, {"text": "c"}]

    assert NoReranker().rerank("query", chunks, top_n=2) == chunks[:2]


def test_context_format_includes_source_markers():
    context = format_context_for_llm(
        [
            {
                "title": "Dog cognition",
                "heading": "Results",
                "doi": "10.123/example",
                "text": "Dogs used the emotional cue.",
            }
        ]
    )

    assert "[Source 1]" in context
    assert "Dog cognition" in context
    assert "Dogs used the emotional cue." in context


def test_prompt_marks_conversation_history_as_non_evidence():
    prompt = build_prompt(
        query="And what about puppies?",
        context="[Source 1] Puppy behaviour\nPuppies respond to social cues.",
        conversation_history=(
            "User: What evidence exists that dogs use human emotional expressions?\n"
            "Assistant: Dogs use human emotional expressions as social cues."
        ),
    )

    assert "Recent conversation:" in prompt
    assert "Do not treat it as scientific evidence" in prompt
    assert "Scientific paper context:" in prompt
    assert "And what about puppies?" in prompt


def test_load_settings_reads_chroma_host(monkeypatch):
    monkeypatch.setenv("CHROMA_HOST", "example.chromadb.dev")
    monkeypatch.setenv("CHROMA_PORT", "443")
    monkeypatch.setenv("CHROMA_SSL", "true")

    settings = load_settings()

    assert settings.chroma_host == "example.chromadb.dev"
    assert settings.chroma_port == 443
    assert settings.chroma_ssl is True


def test_load_settings_uses_final_deployment_profile(monkeypatch):
    for name in [
        "RAG_PIPELINE",
        "ALLOW_RETRIEVAL_OVERRIDES",
        "RETRIEVE_TOP_K",
        "CONTEXT_TOP_N",
        "RERANKER_PROVIDER",
    ]:
        monkeypatch.delenv(name, raising=False)

    settings = load_settings()

    assert settings.rag_pipeline == FINAL_RAG_DEPLOYMENT_PROFILE.name
    assert settings.allow_retrieval_overrides is False
    assert settings.retrieve_top_k == FINAL_RAG_DEPLOYMENT_PROFILE.retrieve_top_k
    assert settings.context_top_n == FINAL_RAG_DEPLOYMENT_PROFILE.context_top_n
    assert settings.reranker_provider == FINAL_RAG_DEPLOYMENT_PROFILE.reranker_provider

