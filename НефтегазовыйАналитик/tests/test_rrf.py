from oil_analyst.rag.retrieval import BM25Index, HybridRetriever


class Embeddings:
    def embed_query(self, query): return [1.0]

class Store:
    def __init__(self, docs): self.docs = docs
    def search(self, vector, limit): return self.docs[:limit]


def test_rrf_rewards_document_present_in_both_lists(make_document):
    shared = make_document("oil demand", "shared", score=0.9)
    dense_only = make_document("other", "dense", score=0.8)
    retriever = HybridRetriever(Embeddings(), Store([dense_only, shared]), BM25Index([shared]), final_k=2)
    result = retriever.retrieve("oil demand")
    assert result[0].metadata.chunk_id == "shared"


def test_reranker_is_called_and_reorders(make_document):
    docs = [make_document("first", "a"), make_document("second", "b")]
    calls = []
    def rerank(query, candidates): calls.append(len(candidates)); return [0.1, 0.9]
    retriever = HybridRetriever(Embeddings(), Store(docs), BM25Index([]), final_k=2, reranker=rerank)
    assert retriever.retrieve("query")[0].metadata.chunk_id == "b"
    assert calls == [2]


def test_supply_section_is_prioritized_for_russian_query(make_document):
    demand = make_document("oil demand", "d", score=0.9)
    demand.metadata.section_title = "World Oil Demand"
    supply = make_document("oil supply", "s", score=0.2)
    supply.metadata.section_title = "World Oil Supply"
    retriever = HybridRetriever(Embeddings(), Store([]), BM25Index([]), final_k=2)
    result = retriever.fuse_and_rerank("о мировом предложении нефти", [demand, supply], [])
    assert result[0].metadata.chunk_id == "s"
