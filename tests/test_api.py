from __future__ import annotations

import json

from tests.conftest import API_KEY, SAMPLE_TEXT, upload_sample


def test_health(client):
    response = client.get("/api/health")
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert payload["storage"]["backend"] == "memory"
    assert payload["llm"]["configured"] is True


def test_upload_and_list(client):
    response = upload_sample(client)
    assert response.status_code == 200
    payload = response.json()
    assert payload["duplicate"] is False
    assert payload["chunk_count"] > 0

    listing = client.get("/api/documents").json()
    assert listing["total"] == 1
    assert listing["items"][0]["filename"] == "差旅报销制度.md"


def test_duplicate_upload_is_skipped(client):
    first = upload_sample(client).json()
    second = upload_sample(client).json()
    assert second["duplicate"] is True
    assert second["document"]["id"] == first["document"]["id"]


def test_upload_requires_api_key(client):
    response = client.post(
        "/api/documents",
        files={"file": ("a.md", b"hello", "text/markdown")},
    )
    assert response.status_code == 401


def test_unsupported_file_type(client):
    response = client.post(
        "/api/documents",
        files={"file": ("a.exe", b"binary", "application/octet-stream")},
        headers={"X-API-Key": API_KEY},
    )
    assert response.status_code == 400


def test_chat_returns_answer_with_citations(client):
    upload_sample(client)
    response = client.post("/api/chat", json={"question": "差旅报销流程是什么？"})
    assert response.status_code == 200
    payload = response.json()
    assert payload["answer"]
    assert payload["citations"]
    assert payload["citations"][0]["filename"] == "差旅报销制度.md"
    assert payload["usage"]["total_tokens"] > 0


def test_chat_caches_repeated_question(client):
    upload_sample(client)
    first = client.post("/api/chat", json={"question": "住宿标准是多少？"}).json()
    second = client.post("/api/chat", json={"question": "住宿标准是多少？"}).json()
    assert first["cache_hit"] is False
    assert second["cache_hit"] is True
    assert second["usage"]["total_tokens"] == 0


def test_chat_with_empty_tenant_returns_no_context(client):
    upload_sample(client)
    response = client.post(
        "/api/chat",
        json={"question": "差旅报销流程是什么？"},
        headers={"X-Tenant-Id": "empty-tenant"},
    )
    assert response.status_code == 200
    assert "无法回答" in response.json()["answer"]


def test_delete_document(client):
    document_id = upload_sample(client).json()["document"]["id"]
    response = client.delete(
        f"/api/documents/{document_id}", headers={"X-API-Key": API_KEY}
    )
    assert response.status_code == 200
    assert client.get("/api/documents").json()["total"] == 0
    assert (
        client.delete(
            f"/api/documents/{document_id}", headers={"X-API-Key": API_KEY}
        ).status_code
        == 404
    )


def test_stream_chat_emits_sources_tokens_done(client):
    upload_sample(client)
    response = client.post(
        "/api/chat/stream", json={"question": "住宿标准是多少？", "use_cache": False}
    )
    assert response.status_code == 200
    events = [
        json.loads(line[len("data: ") :])
        for line in response.text.splitlines()
        if line.startswith("data: ")
    ]
    types = [event["type"] for event in events]
    assert types[0] == "sources"
    assert "token" in types
    assert types[-1] == "done"
    assert "".join(e["text"] for e in events if e["type"] == "token").strip()


def test_stats_aggregates_queries(client):
    upload_sample(client)
    client.post("/api/chat", json={"question": "住宿标准是多少？"})
    stats = client.get("/api/stats").json()
    assert stats["documents"] == 1
    assert stats["chunks"] > 0
    assert stats["queries"] == 1


def test_parent_child_retrieval_returns_context(container):
    """父块回溯：命中子块后，送进 prompt 的应该是更完整的父块文本。"""
    import anyio

    async def run():
        await container.ingestion.ingest("差旅报销制度.md", SAMPLE_TEXT.encode(), "default")
        sources = await container.retriever.retrieve("住宿标准", "default", top_k=3)
        assert sources
        assert any(source.context for source in sources)

    anyio.run(run)
