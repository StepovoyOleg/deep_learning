import json
import logging
import re
from datetime import date, timedelta
from typing import Callable, NotRequired, TypedDict

from .citations import report_source, web_source
from .forecasting import PriceDataProvider, forecast_oil_price
from .interfaces import TextGenerator, WebSearchProvider
from .models import AnalystResponse, ForecastResult, GeneratedAnswer, GeneratedNarrative, QueryPlan, RetrievedDocument, Source, WebResult
from .routing import analyze_query
from .validation import validate_response

logger = logging.getLogger(__name__)


def prioritize_web_results(results: list[WebResult], current_query: bool,
                           today: date | None = None) -> list[WebResult]:
    today = today or date.today()
    selected = list(results)
    if current_query:
        cutoff = today - timedelta(days=548)
        selected = [item for item in selected if item.published_date is None or item.published_date >= cutoff]
    return sorted(
        selected,
        key=lambda item: (item.published_date is not None, item.published_date or date.min, item.score),
        reverse=True,
    )


def answer_quality_issue(answer: str, query: str, reports: list[RetrievedDocument],
                         web: list[WebResult]) -> str | None:
    words = re.findall(r"\w+", answer, re.UNICODE)
    sentences = re.findall(r"[.!?](?:\s|$)", answer.strip())
    if len(words) < 8 or len(sentences) < 2:
        return "Ответ должен содержать 2–4 коротких аналитических предложения, а не одно число или фразу."
    denial = any(phrase in answer.casefold() for phrase in (
        "данных нет", "информация отсутствует", "отсутствует информация", "нет информации",
    ))
    if denial:
        q = query.casefold()
        requested = "supply" if "предлож" in q or "supply" in q else ("demand" if "спрос" in q or "demand" in q else None)
        if requested and any(requested in (doc.metadata.section_title or "").casefold() for doc in reports):
            return "В context есть частично релевантные факты: изложи их и явно ограничь scope вместо полного отрицания."
    if reports and web:
        lowered = answer.casefold()
        if "по загруженным отч" not in lowered or "по текущим web-источник" not in lowered:
            return "Раздели временные слои явными фразами «По загруженным отчётам...» и «По текущим web-источникам...»."
    return None


def build_generation_context(reports: list[RetrievedDocument], web: list[WebResult],
                             forecast: ForecastResult | None) -> str:
    blocks: list[str] = []
    for index, document in enumerate(reports, 1):
        blocks.append(f"SOURCE {index} [REPORT]\n{document.text}")
    offset = len(blocks)
    for index, result in enumerate(web, offset + 1):
        published = result.published_date.isoformat() if result.published_date else "unknown"
        blocks.append(f"SOURCE {index} [WEB]\nTitle: {result.title}\nPublished: {published}\nText: {result.content}")
    if forecast:
        blocks.append("SOURCE FORECAST [BACKEND MODEL]\n" + json.dumps(
            forecast.model_dump(mode="json"), ensure_ascii=False
        ))
    return "\n\n".join(blocks)


class AgentState(TypedDict):
    query: str
    plan: NotRequired[QueryPlan]
    reports: NotRequired[list[RetrievedDocument]]
    web: NotRequired[list[WebResult]]
    forecast: NotRequired[ForecastResult | None]
    sources: NotRequired[list[Source]]
    response: NotRequired[AnalystResponse]
    route: NotRequired[list[str]]
    web_unavailable: NotRequired[bool]


class AnalystAgent:
    def __init__(self, retriever: Callable[..., list[RetrievedDocument]],
                 web_search: WebSearchProvider | None = None,
                 price_provider: PriceDataProvider | None = None,
                 generator: TextGenerator | None = None,
                 min_rag_score: float = 0.0):
        self.retriever, self.web_search = retriever, web_search
        self.price_provider, self.generator, self.min_rag_score = price_provider, generator, min_rag_score
        self.graph = self._build_graph()

    def _build_graph(self):
        from langgraph.graph import END, START, StateGraph
        graph = StateGraph(AgentState)
        graph.add_node("analyze", self._analyze)
        graph.add_node("retrieve_reports", self._retrieve)
        graph.add_node("search_web", self._web)
        graph.add_node("forecast", self._forecast)
        graph.add_node("generate", self._generate)
        graph.add_node("validate", self._validate)
        graph.add_edge(START, "analyze")
        graph.add_conditional_edges(
            "analyze",
            lambda s: "generate" if not s["plan"].in_scope else (
                "forecast" if s["plan"].needs_forecast else ("search_web" if s["plan"].web_only else "retrieve_reports")
            ),
        )
        graph.add_conditional_edges("retrieve_reports", self._after_reports,
                                    {"web": "search_web", "forecast": "forecast", "generate": "generate"})
        graph.add_conditional_edges("search_web", lambda s: "forecast" if s["plan"].needs_forecast else "generate")
        graph.add_edge("forecast", "generate"); graph.add_edge("generate", "validate"); graph.add_edge("validate", END)
        return graph.compile()

    def _analyze(self, state: AgentState) -> dict:
        return {"plan": analyze_query(state["query"]), "route": ["query_analysis"]}

    def _retrieve(self, state: AgentState) -> dict:
        filters = state["plan"].metadata_filters
        try:
            reports = self.retriever(state["query"], filters)
        except TypeError:
            reports = self.retriever(state["query"])
        return {"reports": reports, "route": state["route"] + ["rag_retrieval"]}

    def _after_reports(self, state: AgentState) -> str:
        plan, reports = state["plan"], state.get("reports", [])
        sufficient = bool(reports) and max(d.score for d in reports) >= self.min_rag_score
        if plan.needs_web or not sufficient: return "web"
        if plan.needs_forecast: return "forecast"
        return "generate"

    def _web(self, state: AgentState) -> dict:
        unavailable = self.web_search is None
        try:
            results = self.web_search.search(state["query"]) if self.web_search else []
            results = prioritize_web_results(results, state["plan"].needs_web)
        except Exception as exc:
            logger.warning("Web search unavailable: %s", exc)
            results, unavailable = [], True
        return {"web": results, "web_unavailable": unavailable,
                "route": state["route"] + ["web_search"]}

    def _forecast(self, state: AgentState) -> dict:
        plan = state["plan"]
        result = None
        if self.price_provider and plan.instrument:
            result = forecast_oil_price(self.price_provider, plan.instrument, plan.horizon or 3)
        return {"forecast": result, "route": state["route"] + ["forecast"]}

    def _generate(self, state: AgentState) -> dict:
        plan, reports, web = state["plan"], state.get("reports", []), state.get("web", [])
        sources = [report_source(x) for x in reports] + [web_source(x) for x in web]
        if not plan.in_scope:
            answer = "Запрос вне моей компетенции. Я специализируюсь на нефтегазовом рынке."
        elif not reports and not web and not state.get("forecast"):
            answer = "Недостаточно подтверждённых данных для ответа. Загрузите отраслевой PDF или настройте web search/источник цен."
        elif self.generator:
            context = build_generation_context(reports, web, state.get("forecast"))
            prompt = ("Ты старший аналитик нефтегазового рынка. Используй только переданный контекст. "
                      "Перед выводом о нехватке данных проверь все SOURCE-блоки. Если релевантный факт явно присутствует, используй его. "
                      "Если доступен частичный ответ, сообщи найденные факты, точно обозначь их scope и не утверждай, что информации нет полностью. "
                      "Не придумывай числа или источники; отличай report, web и forecast. Дай содержательный ответ строго в 2–4 предложениях, если данных достаточно. "
                      "Не называй старые web-факты текущими. Если есть и REPORT, и WEB, раздели ответ фразами «По загруженным отчётам...» и «По текущим web-источникам...». "
                      "Верни только answer и uncertainty; не генерируй URL, source_refs, страницы, имена документов или значения прогноза.\nЗапрос: " + state["query"] +
                      "\n\nCONTEXT:\n" + context)
            generated = None
            try:
                raw = self.generator.generate(prompt)
            except Exception as exc:
                logger.warning("LLM unavailable; using grounded fallback: %s", exc)
                raw = ""
            for attempt in range(2):
                try:
                    narrative = GeneratedNarrative.model_validate_json(raw)
                    quality_issue = answer_quality_issue(narrative.answer, state["query"], reports, web)
                    if quality_issue:
                        raise ValueError(quality_issue)
                    generated = GeneratedAnswer(
                        answer=narrative.answer,
                        uncertainty=narrative.uncertainty,
                        source_refs=[source.citation for source in sources],
                    )
                    break
                except Exception as exc:
                    if attempt == 0 and raw:
                        try:
                            raw = self.generator.generate(prompt + "\nИсправь предыдущий ответ и верни только валидный JSON. Ошибка: " + str(exc))
                        except Exception:
                            raw = ""
                    else:
                        logger.warning("Structured generation failed after one repair: %s", exc)
                        break
            if generated:
                answer, uncertainty = generated.answer, generated.uncertainty
            else:
                if state.get("forecast"):
                    forecast = state["forecast"]
                    rows = [f"- {p.period}: {p.value:.2f} USD/bbl (80%: {p.lower_bound:.2f}–{p.upper_bound:.2f})"
                            for p in forecast.forecast]
                    answer = f"Статистический прогноз {forecast.instrument}:\n\n" + "\n".join(rows) + "\n\n" + forecast.interpretation
                else:
                    snippets = [d.text[:500] for d in reports[:3]] + [w.content[:500] for w in web[:3]]
                    answer = "Подтверждённый контекст:\n\n" + "\n\n".join(f"- {x}" for x in snippets)
                uncertainty = "Ответ основан на прямых выдержках из доступных источников."
        else:
            if state.get("forecast"):
                forecast = state["forecast"]
                rows = [f"- {p.period}: {p.value:.2f} USD/bbl (80%: {p.lower_bound:.2f}–{p.upper_bound:.2f})"
                        for p in forecast.forecast]
                answer = f"Статистический прогноз {forecast.instrument}:\n\n" + "\n".join(rows) + "\n\n" + forecast.interpretation
            else:
                snippets = [d.text[:500] for d in reports[:3]] + [w.content[:500] for w in web[:3]]
                answer = "Подтверждённый контекст:\n\n" + "\n\n".join(f"- {x}" for x in snippets)
        warnings = ["Web search unavailable: TAVILY_API_KEY is not configured."] if state.get("web_unavailable") else []
        response = AnalystResponse(answer=answer, sources=sources, forecast=state.get("forecast"),
                                   route=state["route"] + ["answer_generation"], warnings=warnings,
                                   metadata_filters=plan.metadata_filters,
                                   uncertainty=locals().get("uncertainty"))
        return {"sources": sources, "response": response, "route": response.route}

    def _validate(self, state: AgentState) -> dict:
        response = validate_response(state["response"], state.get("sources", []))
        response.route.append("validation")
        return {"response": response, "route": response.route}

    def invoke(self, query: str) -> AnalystResponse:
        return self.graph.invoke({"query": query})["response"]
