from oil_analyst.routing import analyze_query


def test_routes_forecast():
    plan = analyze_query("Спрогнозируй Brent на следующие 3 месяца")
    assert plan.in_scope and plan.needs_forecast and plan.horizon == 3


def test_routes_out_of_scope():
    assert not analyze_query("Напиши рецепт борща").in_scope


def test_routes_current_to_web():
    assert analyze_query("Последние новости OPEC сегодня").needs_web
