from datetime import date

from oil_analyst.models import WebResult
from oil_analyst.orchestration import AnalystAgent, build_generation_context, prioritize_web_results


class SequenceGenerator:
    def __init__(self, responses): self.responses, self.calls = iter(responses), 0
    def generate(self, prompt): self.calls += 1; return next(self.responses)


def test_valid_structured_output(make_document):
    generator = SequenceGenerator(['{"answer":"OPEC expects demand growth in 2026. The supplied context supports this assessment.","uncertainty":null}'])
    response = AnalystAgent(lambda *_: [make_document("oil demand", score=1)], generator=generator).invoke("OPEC oil demand")
    assert response.answer.startswith("OPEC expects") and generator.calls == 1
    assert response.validation["valid"]


def test_invalid_json_gets_one_repair(make_document):
    generator = SequenceGenerator(["not-json", '{"answer":"The repaired answer uses the supplied evidence. Its scope remains explicitly limited.","uncertainty":"limited"}'])
    response = AnalystAgent(lambda *_: [make_document("oil demand", score=1)], generator=generator).invoke("OPEC oil demand")
    assert response.answer.startswith("The repaired") and response.uncertainty == "limited"
    assert generator.calls == 2


def test_second_invalid_json_falls_back(make_document):
    generator = SequenceGenerator(["bad", "still bad"])
    response = AnalystAgent(lambda *_: [make_document("oil demand", score=1)], generator=generator).invoke("OPEC oil demand")
    assert response.answer.startswith("Подтверждённый контекст") and generator.calls == 2
    assert response.uncertainty != "Structured LLM output unavailable"


def test_llm_cannot_add_fabricated_source(make_document):
    generator = SequenceGenerator([
        '{"answer":"Grounded","uncertainty":null,"source_refs":["https://fabricated.invalid"]}',
        '{"answer":"The repaired answer uses only real evidence. Fabricated provenance is not included.","uncertainty":null}',
    ])
    response = AnalystAgent(lambda *_: [make_document("oil demand", score=1)], generator=generator).invoke("OPEC oil demand")
    assert response.answer.startswith("The repaired")
    assert generator.calls == 2
    assert all("fabricated.invalid" not in source.citation for source in response.sources)


def test_source_refs_come_only_from_tool_results(make_document):
    generator = SequenceGenerator(['{"answer":"The answer is grounded in the report context. Backend sources remain canonical.","uncertainty":null}'])
    response = AnalystAgent(lambda *_: [make_document("oil demand", score=1)], generator=generator).invoke("OPEC oil demand")
    assert [source.citation for source in response.sources]
    assert all(source.kind == "report" for source in response.sources)


def test_generation_context_uses_readable_source_blocks(make_document):
    document = make_document(
        "Organization: OPEC\nSection: World Oil Supply\nPage: 53\n\nThe 2026 forecast is 8.8 mb/d.",
        page=53,
    )
    context = build_generation_context([document], [], None)
    assert context.startswith("SOURCE 1 [REPORT]")
    assert "Section: World Oil Supply" in context
    assert "The 2026 forecast is 8.8 mb/d." in context


def test_short_numeric_rag_answer_is_repaired(make_document):
    generator = SequenceGenerator([
        '{"answer":"0.8 mb/d","uncertainty":null}',
        '{"answer":"OPEC expects world oil demand to grow by 0.8 mb/d in 2026. This indicates continued expansion, led by the supplied report context.","uncertainty":null}',
    ])
    response = AnalystAgent(lambda *_: [make_document("World Oil Demand: growth 0.8 mb/d", score=1)], generator=generator).invoke("OPEC demand")
    assert generator.calls == 2
    assert "continued expansion" in response.answer


def test_partial_supply_context_replaces_false_denial(make_document):
    document = make_document("DoC NGLs rise by 0.1 mb/d to 8.8 mb/d", score=1)
    document.metadata.section_title = "World Oil Supply"
    generator = SequenceGenerator([
        '{"answer":"В контексте данных нет. Информация о предложении полностью отсутствует.","uncertainty":null}',
        '{"answer":"OPEC forecasts DoC NGLs growth of 0.1 mb/d to 8.8 mb/d. This figure covers DoC NGLs, not total global supply.","uncertainty":null}',
    ])
    response = AnalystAgent(lambda *_: [document], generator=generator).invoke("Что сказано о предложении нефти?")
    assert generator.calls == 2
    assert "not total global supply" in response.answer


def test_fresh_web_results_are_preferred_and_2024_is_excluded():
    results = [
        WebResult(title="old", url="https://example.com/old", content="2024 fact", published_date=date(2024, 6, 1), score=1),
        WebResult(title="new", url="https://example.com/new", content="2026 fact", published_date=date(2026, 7, 1), score=0.5),
        WebResult(title="unknown", url="https://example.com/unknown", content="undated", score=0.9),
    ]
    selected = prioritize_web_results(results, current_query=True, today=date(2026, 8, 18))
    assert [item.title for item in selected] == ["new", "unknown"]
    context = build_generation_context([], selected, None)
    assert "2026-07-01" in context and "2024 fact" not in context


def test_rag_web_answer_is_repaired_until_time_layers_are_explicit(make_document):
    document = make_document("Report forecast for 2026", score=1)
    web = WebResult(title="Current market", url="https://example.com/current",
                    content="Current market conditions", published_date=date(2026, 8, 1))
    generator = SequenceGenerator([
        '{"answer":"The report forecast and current market conditions differ. The comparison uses both contexts.","uncertainty":null}',
        '{"answer":"По загруженным отчётам прогноз относится к 2026 году. По текущим web-источникам рыночная ситуация изменилась позднее.","uncertainty":null}',
    ])
    response = AnalystAgent(lambda *_: [document], web_search=type("Web", (), {"search": lambda self, query: [web]})(),
                            generator=generator).invoke("Compare the report with the current oil market")
    assert generator.calls == 2
    assert "По загруженным отчётам" in response.answer
    assert "По текущим web-источникам" in response.answer
