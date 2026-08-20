from oil_analyst.adapters import QdrantVectorStore
from oil_analyst.models import MetadataFilters


def test_qdrant_filters_by_nested_organization(make_document):
    from qdrant_client import QdrantClient
    store = QdrantVectorStore.__new__(QdrantVectorStore)
    store.client, store.collection = QdrantClient(":memory:"), "filtered"
    opec = make_document("OPEC demand", "opec")
    eia = make_document("EIA demand", "eia").model_copy(deep=True)
    eia.metadata.organization = "EIA"
    store.upsert([opec, eia], [[1.0, 0.0], [1.0, 0.0]])
    result = store.search([1.0, 0.0], 5, MetadataFilters(organization="EIA"))
    assert len(result) == 1 and result[0].metadata.organization == "EIA"
