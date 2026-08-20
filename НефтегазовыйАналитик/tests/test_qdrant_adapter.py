from oil_analyst.adapters import QdrantVectorStore


def test_qdrant_adapter_upsert_is_idempotent_in_memory(make_document):
    from qdrant_client import QdrantClient
    store = QdrantVectorStore.__new__(QdrantVectorStore)
    store.client, store.collection = QdrantClient(":memory:"), "test"
    doc = make_document("oil demand", "stable-id")
    store.upsert([doc], [[1.0, 0.0]])
    store.upsert([doc], [[1.0, 0.0]])
    assert store.client.count("test", exact=True).count == 1
    assert store.search([1.0, 0.0], 1)[0].metadata.chunk_id == "stable-id"
