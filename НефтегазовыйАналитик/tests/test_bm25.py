from oil_analyst.rag.retrieval import BM25Index


def test_bm25_ranks_lexical_match_first(make_document):
    docs = [make_document("world oil demand growth", "a"), make_document("natural gas storage", "b")]
    result = BM25Index(docs).search("oil demand", 2)
    assert result[0].metadata.chunk_id == "a" and result[0].score > result[1].score
