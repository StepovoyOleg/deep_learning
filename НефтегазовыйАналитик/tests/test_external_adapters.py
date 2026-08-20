from oil_analyst.adapters import OllamaGenerator, TavilySearch


def test_tavily_adapter_keeps_allowlisted_domains():
    adapter = TavilySearch.__new__(TavilySearch)
    adapter.domains = ["eia.gov"]
    adapter.timeout, adapter.max_attempts, adapter.backoff_seconds = 3, 3, 0
    class Client:
        def search(self, *args, **kwargs):
            return {"results": [
                {"title": "EIA", "url": "https://www.eia.gov/test", "content": "official", "score": 1},
                {"title": "SEO", "url": "https://example.com/test", "content": "other", "score": 1},
            ]}
    adapter.client = Client()
    assert [str(item.url) for item in adapter.search("oil")] == ["https://www.eia.gov/test"]


def test_tavily_retry_is_bounded(monkeypatch):
    adapter = TavilySearch.__new__(TavilySearch)
    adapter.domains, adapter.timeout = ["eia.gov"], 3
    adapter.max_attempts, adapter.backoff_seconds = 3, 0.01
    calls = []
    class Client:
        def search(self, *args, **kwargs):
            calls.append(kwargs)
            raise ConnectionError("transient")
    adapter.client = Client()
    monkeypatch.setattr("oil_analyst.adapters.time.sleep", lambda seconds: None)
    import pytest
    with pytest.raises(ConnectionError, match="transient"):
        adapter.search("oil")
    assert len(calls) == 3
    assert all(call["timeout"] == 3 for call in calls)


def test_ollama_adapter_posts_structured_schema(monkeypatch):
    calls = []
    class Response:
        def raise_for_status(self): pass
        def json(self): return {"message": {"content": '{"answer":"ok","uncertainty":null}'}}
    class Client:
        def __init__(self, **kwargs): calls.append(kwargs)
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def post(self, url, json): calls.append((url, json)); return Response()
    import httpx
    monkeypatch.setattr(httpx, "Client", Client)
    raw = OllamaGenerator("http://localhost:11434", "qwen3:4b").generate("prompt")
    assert '"answer":"ok"' in raw and calls[0]["trust_env"] is False
    assert calls[1][1]["format"]["type"] == "object"
    assert "source_refs" not in calls[1][1]["format"]["properties"]
