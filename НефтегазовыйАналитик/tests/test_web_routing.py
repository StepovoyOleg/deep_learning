from oil_analyst.routing import analyze_query


def test_latest_news_is_web_only():
    plan = analyze_query("Какие последние новости OPEC+ сегодня?")
    assert plan.needs_web and plan.web_only and not plan.needs_forecast


def test_comparison_is_rag_plus_web_not_forecast():
    plan = analyze_query("Сравни прогноз OPEC/EIA с текущей ситуацией на нефтяном рынке")
    assert plan.needs_web and not plan.web_only and not plan.needs_forecast
    assert not plan.metadata_filters.active()
