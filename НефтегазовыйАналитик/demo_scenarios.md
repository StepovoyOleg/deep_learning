# Demo scenarios

| Scenario | Query | Expected route |
|---|---|---|
| RAG only | «Как OPEC оценивает рост мирового спроса в загруженном MOMR?» | analysis → RAG → generation → validation |
| Web | «Какова текущая цена Brent сегодня?» | analysis → RAG → web → generation → validation |
| RAG + Web | «Сопоставь прогноз IEA из отчёта с последними событиями OPEC+» | analysis → RAG → web → generation → validation |
| Forecast | «Спрогнозируй Brent на 3 месяца» | analysis → RAG → forecast → generation → validation |
| Out of scope | «Напиши рецепт борща» | analysis → generation → validation |
