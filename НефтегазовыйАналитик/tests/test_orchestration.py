from oil_analyst.orchestration import AnalystAgent


def test_out_of_scope_graph_route():
    response = AnalystAgent(lambda _: []).invoke("Напиши рецепт борща")
    assert response.route == ["query_analysis", "answer_generation", "validation"]


def test_rag_graph_route(make_document):
    response = AnalystAgent(lambda _: [make_document("OPEC oil demand outlook", score=1.0)]).invoke("Что OPEC ожидает по спросу на нефть?")
    assert "rag_retrieval" in response.route and response.sources[0].kind == "report"


def test_forecast_graph_route():
    class Provider:
        def load(self, instrument):
            from oil_analyst.forecasting import EiaBrentDataProvider
            return EiaBrentDataProvider("data/prices/RBRTEd.xls").load(instrument)
    response = AnalystAgent(lambda _: [], price_provider=Provider()).invoke("Спрогнозируй Brent на 3 месяца")
    assert response.forecast is not None and "forecast" in response.route and "rag_retrieval" not in response.route
    assert "2026" in response.answer and "80%" in response.answer


def test_web_failure_degrades_gracefully():
    class FailingWeb:
        def search(self, query):
            raise ConnectionError("temporary Tavily failure")
    response = AnalystAgent(lambda *_: [], web_search=FailingWeb()).invoke(
        "What is the latest OPEC oil market news today?"
    )
    assert "web_search" in response.route
    assert response.validation["valid"]
    assert response.warnings
