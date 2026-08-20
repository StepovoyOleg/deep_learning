import re

from .models import MetadataFilters, QueryPlan

DOMAIN = {"oil", "gas", "brent", "wti", "urals", "opec", "eia", "iea", "нефт", "газ", "добыч", "нпз", "трубопровод"}
CURRENT = {"today", "current", "latest", "now", "сегодня", "текущ", "актуальн", "последн", "цена"}
FORECAST = {"forecast", "predict", "спрогноз", "прогнозируй"}
MONTHS = {
    "january": 1, "январ": 1, "february": 2, "феврал": 2, "march": 3, "март": 3,
    "april": 4, "апрел": 4, "may": 5, "мая": 5, "май": 5, "june": 6, "июн": 6,
    "july": 7, "июл": 7, "august": 8, "август": 8, "september": 9, "сентябр": 9,
    "october": 10, "октябр": 10, "november": 11, "ноябр": 11, "december": 12, "декабр": 12,
}


def extract_metadata_filters(query: str) -> MetadataFilters:
    q = query.casefold()
    has_opec = "opec" in q or "опек" in q
    has_eia = "eia" in q or "мэа сша" in q
    organization = None if has_opec and has_eia else ("OPEC" if has_opec else ("EIA" if has_eia else None))
    year_match = re.search(r"\b(20\d{2})\b", q)
    month = next((number for stem, number in MONTHS.items() if stem in q), None)
    report_date = f"{year_match.group(1)}-{month:02d}" if year_match and month else None
    report_name = "OPEC Monthly Oil Market Report" if "momr" in q else ("EIA Short-Term Energy Outlook" if "steo" in q else None)
    return MetadataFilters(organization=organization, report_date=report_date, report_name=report_name)


def analyze_query(query: str) -> QueryPlan:
    q = query.casefold()
    in_scope = any(term in q for term in DOMAIN)
    needs_forecast = any(term in q for term in FORECAST)
    needs_web = any(term in q for term in CURRENT)
    web_only = needs_web and not any(term in q for term in ("сравни", "compare", "сопостав"))
    instrument = next((x for x in ("Brent", "WTI", "Urals") if x.casefold() in q), None)
    match = re.search(r"(\d+)\s*(?:month|months|месяц|месяца|месяцев)", q)
    horizon = int(match.group(1)) if match else (3 if needs_forecast else None)
    reasons = [] if in_scope else ["Запрос вне нефтегазовой специализации"]
    return QueryPlan(in_scope=in_scope, needs_web=needs_web, needs_forecast=needs_forecast, web_only=web_only,
                     instrument=instrument, horizon=horizon, reasons=reasons,
                     metadata_filters=extract_metadata_filters(query))
