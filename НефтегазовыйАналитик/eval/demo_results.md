# Demo check results

Raw smoke-check output from the existing production pipeline. No automatic PASS/FAIL evaluation is applied.

## 1. RAG

**Query**

Как OPEC оценивает мировой спрос на нефть в 2026 году?

**Execution time:** 102.00 seconds

**Route**

```json
[
  "query_analysis",
  "rag_retrieval",
  "answer_generation",
  "validation"
]
```

**Answer**

По отчётам OPEC за июнь и июль 2026 года прогнозируемый рост мирового спроса на нефть в 2026 году составляет 1,0 млн баррелей в сутки (июнь) и 0,8 млн баррелей в сутки (июль). Корректировки связаны с динамикой спроса в регионах Азии и Африки. Абсолютный спрос в 2026 году по последнему отчёту составляет 107,9 млн баррелей в сутки.

**Uncertainty**

low

**Sources**

```json
[
  {
    "kind": "report",
    "citation": "[Report: OPEC Monthly Oil Market Report, May 2026, p. 41]",
    "url": null
  },
  {
    "kind": "report",
    "citation": "[Report: OPEC Monthly Oil Market Report, July 2026, p. 41]",
    "url": null
  },
  {
    "kind": "report",
    "citation": "[Report: OPEC Monthly Oil Market Report, June 2026, p. 43]",
    "url": null
  },
  {
    "kind": "report",
    "citation": "[Report: OPEC Monthly Oil Market Report, July 2026, p. 43]",
    "url": null
  },
  {
    "kind": "report",
    "citation": "[Report: OPEC Monthly Oil Market Report, June 2026, p. 87]",
    "url": null
  },
  {
    "kind": "report",
    "citation": "[Report: OPEC Monthly Oil Market Report, July 2026, p. 41]",
    "url": null
  }
]
```

**Metadata filters**

```json
{
  "organization": "OPEC"
}
```

**Forecast result**

```json
null
```

## 2. RAG_METADATA

**Query**

Что OPEC писал о мировом предложении нефти в июльском отчёте 2026 года?

**Execution time:** 132.11 seconds

**Route**

```json
[
  "query_analysis",
  "rag_retrieval",
  "answer_generation",
  "validation"
]
```

**Answer**

По загруженным отчётам OPEC в июльском отчёте 2026 года прогнозируется рост мирового предложения нефтепродуктов неучастников DoC на 0,6 млн баррелей в сутки до уровня 54,8 млн баррелей в сутки в 2026 году, с основными факторами роста — Бразилией, США, Канадой и Аргентиной. Доля нефтегазовых жидкостей и нефтегазовых нефтей в DoC также прогнозируется на уровне 8,8 млн баррелей в сутки в 2026 году.

**Uncertainty**

low

**Sources**

```json
[
  {
    "kind": "report",
    "citation": "[Report: OPEC Monthly Oil Market Report, July 2026, p. 53]",
    "url": null
  },
  {
    "kind": "report",
    "citation": "[Report: OPEC Monthly Oil Market Report, July 2026, p. 55]",
    "url": null
  },
  {
    "kind": "report",
    "citation": "[Report: OPEC Monthly Oil Market Report, July 2026, p. 63]",
    "url": null
  },
  {
    "kind": "report",
    "citation": "[Report: OPEC Monthly Oil Market Report, July 2026, p. 59]",
    "url": null
  },
  {
    "kind": "report",
    "citation": "[Report: OPEC Monthly Oil Market Report, July 2026, p. 62]",
    "url": null
  },
  {
    "kind": "report",
    "citation": "[Report: OPEC Monthly Oil Market Report, July 2026, p. 85]",
    "url": null
  }
]
```

**Metadata filters**

```json
{
  "organization": "OPEC",
  "report_date": "2026-07"
}
```

**Forecast result**

```json
null
```

## 3. WEB

**Query**

Какие последние значимые новости по OPEC+?

**Execution time:** 48.61 seconds

**Route**

```json
[
  "query_analysis",
  "web_search",
  "answer_generation",
  "validation"
]
```

**Answer**

По загруженным отчётам: OPEC+ страны (8) согласовали увеличение производства на 547 кб/д в сентябре 2025 года, полностью отменяя сокращения, которые были введены в ноябре 2023 года. По текущим web-источникам: OPEC+ страны провели виртуальную встречу 31 мая 2025 года и планируют ежемесячные встречи для регулирования производства, включая решение о производственных уровнях в июле 2025 года.

**Uncertainty**

low

**Sources**

```json
[
  {
    "kind": "web",
    "citation": "[Organization of the Petroleum Exporting Countries, date unavailable, https://www.opec.org/]",
    "url": "https://www.opec.org/"
  },
  {
    "kind": "web",
    "citation": "[Organization of the Petroleum Exporting Countries, date unavailable, https://www.opec.org/pr-detail/1360566-31-may-2025.html]",
    "url": "https://www.opec.org/pr-detail/1360566-31-may-2025.html"
  },
  {
    "kind": "web",
    "citation": "[IEA, date unavailable, https://www.iea.org/reports/oil-market-report-march-2025]",
    "url": "https://www.iea.org/reports/oil-market-report-march-2025"
  },
  {
    "kind": "web",
    "citation": "[EIA forecasts lower oil price in 2025 amid significant ..., date unavailable, https://www.eia.gov/todayinenergy/detail.php?id=64305]",
    "url": "https://www.eia.gov/todayinenergy/detail.php?id=64305"
  },
  {
    "kind": "web",
    "citation": "[IEA, date unavailable, https://www.iea.org/reports/oil-market-report-august-2025]",
    "url": "https://www.iea.org/reports/oil-market-report-august-2025"
  }
]
```

**Metadata filters**

```json
{
  "organization": "OPEC"
}
```

**Forecast result**

```json
null
```

## 4. RAG_WEB

**Query**

Сравни прогноз OPEC/EIA из загруженных отчётов с текущей ситуацией на нефтяном рынке.

**Execution time:** 127.45 seconds

**Route**

```json
[
  "query_analysis",
  "rag_retrieval",
  "web_search",
  "answer_generation",
  "validation"
]
```

**Answer**

Подтверждённый контекст:

- Organization: EIA
Report: EIA Short-Term Energy Outlook
Date: May 2026
Page: 40

..... 0.09 0.08 0.06 0.04 0.03 0.03 0.04 0.04 0.03 0.03 0.03 0.03 0.07 0.04 0.03 Unplanned production outages OPEC total ......................................................................................................... 1.03 1.00 1.00 0.91 3.62 - - - - - - - 0.98 - - Notes: Sources: (a) Differences in the reported historical production data across countries could result in some inconsistencies in the delineat

- Organization: EIA
Report: EIA Short-Term Energy Outlook
Date: May 2026
Page: 3

production capacity, we now expect OPEC’s spare capacity to average 2.5 million b/d in 2027, compared with our previous forecast of 3.8 million b/d. Short-Term Energy Outlook

- Organization: EIA
Report: EIA Short-Term Energy Outlook
Date: May 2026
Page: 40

Forecasts: EIA Short-Term Integrated Forecasting System. (b) OPEC+ total = OPEC members subject to OPEC+ agreements plus Azerbaijan, Bahrain, Brunei, Kazakhstan, Malaysia, Mexico, Oman, Russia, South Sudan, and Sudan. (c) OPEC = Organization of the Petroleum Exporting Countries: Algeria, Congo (Brazzaville), Equatorial Guinea, Gabon, Iran, Iraq, Kuwait, Libya, Nigeria, Saudi Arabia, and Venezuela. (d) Iran, Libya, a

- At the same time, OPEC+ producers reconfirmed their plan to maintain current production quotas through March. In this context, global oil supply is expected to rebound in the coming months as output recovers from the exceptional plunge in January, when extreme winter weather forced the shut-in of over 1 mb/d of output in North America. In addition, prolonged disruptions at Kazakhstan’s key export terminal since November were compounded by a power outage at the country’s largest field last month,

- Total global oil supply rose by 760 kb/d m-o-m, to 108 mb/d in September, as OPEC+ production surged by 1 mb/d led by the Middle East. World oil supply is on track to rise by 3 mb/d to 106.1 mb/d this year and 2.4 mb/d next year. Non-OPEC+ adds 1.6 mb/d and 1.2 mb/d, respectively, led by the US, Brazil, Canada, Guyana and Argentina. OPEC+ adds 1.4 mb/d in 2025 and 1.2 mb/d next year based on the current production agreement. [...] |  | Aug 2025 Supply | Sep 2025 Supply | Sep 2025 vs Target | Sep

- |  | Feb 2026 Supply | Mar 2026 Supply | Mar 2026 vs Target | Mar 2026 Implied Target1 | Sustainable Capacity2 | Eff Spare Cap vs Mar3 |
 ---  ---  --- 
| Algeria | 0.98 | 0.96 | -0.01 | 0.97 | 0.99 | 0.02 |
| Congo | 0.3 | 0.27 | -0.01 | 0.28 | 0.27 | 0 |
| Equatorial Guinea | 0.06 | 0.06 | -0.01 | 0.07 | 0.06 | 0.0 |
| Gabon | 0.2 | 0.19 | 0.01 | 0.18 | 0.22 | 0.04 |
| Iraq | 4.57 | 1.57 | -2.59 | 4.16 | 4.87 |  |
| Kuwait | 2.54 | 1.19 | -1.39 | 2.58 | 2.88 |  |
| Nigeria | 1.31 | 1.35 | -0.1

**Uncertainty**

Ответ основан на прямых выдержках из доступных источников.

**Sources**

```json
[
  {
    "kind": "report",
    "citation": "[Report: EIA Short-Term Energy Outlook, May 2026, p. 40]",
    "url": null
  },
  {
    "kind": "report",
    "citation": "[Report: EIA Short-Term Energy Outlook, May 2026, p. 3]",
    "url": null
  },
  {
    "kind": "report",
    "citation": "[Report: EIA Short-Term Energy Outlook, May 2026, p. 40]",
    "url": null
  },
  {
    "kind": "report",
    "citation": "[Report: EIA Short-Term Energy Outlook, July 2026, p. 36]",
    "url": null
  },
  {
    "kind": "report",
    "citation": "[Report: OPEC Monthly Oil Market Report, July 2026, p. 51]",
    "url": null
  },
  {
    "kind": "report",
    "citation": "[Report: OPEC Monthly Oil Market Report, May 2026, p. 13]",
    "url": null
  },
  {
    "kind": "web",
    "citation": "[IEA, date unavailable, https://www.iea.org/reports/oil-market-report-february-2026]",
    "url": "https://www.iea.org/reports/oil-market-report-february-2026"
  },
  {
    "kind": "web",
    "citation": "[IEA, date unavailable, https://www.iea.org/reports/oil-market-report-october-2025]",
    "url": "https://www.iea.org/reports/oil-market-report-october-2025"
  },
  {
    "kind": "web",
    "citation": "[IEA, date unavailable, https://www.iea.org/reports/oil-market-report-april-2026]",
    "url": "https://www.iea.org/reports/oil-market-report-april-2026"
  },
  {
    "kind": "web",
    "citation": "[EIA forecasts OPEC production will grow in 2022 despite recent production outages in Libya -\r\n\t\tU.S. Energy Information Administration (EIA), date unavailable, http://www.eia.gov/todayinenergy/detail.php?id=51078]",
    "url": "http://www.eia.gov/todayinenergy/detail.php?id=51078"
  },
  {
    "kind": "web",
    "citation": "[EIA forecasts OPEC production will grow in 2022 despite recent production outages in Libya -\r\n\t\tU.S. Energy Information Administration (EIA), date unavailable, https://www.eia.gov/todayinenergy/detail.php?id=51078]",
    "url": "https://www.eia.gov/todayinenergy/detail.php?id=51078"
  }
]
```

**Metadata filters**

```json
{}
```

**Forecast result**

```json
null
```

## 5. FORECAST

**Query**

Спрогнозируй цену Brent на следующие 3 месяца.

**Execution time:** 25.73 seconds

**Route**

```json
[
  "query_analysis",
  "forecast",
  "answer_generation",
  "validation"
]
```

**Answer**

По загруженным отчётам модель прогнозирует цену Brent на 2026-09-01 в 89,69 доллара за баррель (диапазон 80,59–98,79), 2026-10-01 в 90,72 доллара за баррель (диапазон 77,85–103,60) и 2026-11-01 в 88,33 доллара за баррель (диапазон 72,56–104,09). Прогноз не учитывает будущие шоки и решения OPEC+.

**Uncertainty**

Прогноз строится на статистической модели, не включая будущие рыночные шоки и решения OPEC+.

**Sources**

```json
[]
```

**Metadata filters**

```json
{}
```

**Forecast result**

```json
{
  "instrument": "Brent",
  "forecast_horizon": 3,
  "method": "exponential_smoothing",
  "forecast": [
    {
      "period": "2026-09-01",
      "value": 89.69423998169415,
      "lower_bound": 80.59046033871567,
      "upper_bound": 98.79801962467263
    },
    {
      "period": "2026-10-01",
      "value": 90.72237228020282,
      "lower_bound": 77.84768364024657,
      "upper_bound": 103.59706092015907
    },
    {
      "period": "2026-11-01",
      "value": 88.32656122352239,
      "lower_bound": 72.55835234097242,
      "upper_bound": 104.09477010607236
    }
  ],
  "lower_bound": [
    80.59046033871567,
    77.84768364024657,
    72.55835234097242
  ],
  "upper_bound": [
    98.79801962467263,
    103.59706092015907,
    104.09477010607236
  ],
  "metrics": {
    "rmse_in_sample": 7.074034543504972,
    "mape_in_sample_pct": 7.930427150608617
  },
  "interpretation": "Статистический базовый прогноз, не учитывающий будущие шоки и решения OPEC+.",
  "assumptions": [
    "Историческая динамика сохраняет прогностическую ценность",
    "Интервал — ориентировочный 80% диапазон"
  ]
}
```

## 6. OUT_OF_SCOPE

**Query**

Напиши рецепт борща.

**Execution time:** 0.00 seconds

**Route**

```json
[
  "query_analysis",
  "answer_generation",
  "validation"
]
```

**Answer**

Запрос вне моей компетенции. Я специализируюсь на нефтегазовом рынке.

**Uncertainty**

None

**Sources**

```json
[]
```

**Metadata filters**

```json
{}
```

**Forecast result**

```json
null
```
