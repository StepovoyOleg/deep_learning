import re

from .models import AnalystResponse, Source


def validate_response(response: AnalystResponse, available: list[Source]) -> AnalystResponse:
    allowed = {s.citation for s in available}
    unknown = [s.citation for s in response.sources if s.citation not in allowed]
    has_numbers = bool(re.search(r"\b\d+(?:[.,]\d+)?\b", response.answer))
    issues = []
    if response.sources and unknown: issues.append("Ответ содержит источник, которого нет в полученном контексте")
    if has_numbers and not response.sources and not response.forecast: issues.append("Числовые утверждения не подкреплены источником")
    if not response.answer.strip(): issues.append("Пустой ответ")
    response.validation = {"valid": not issues, "issues": issues, "citations_verified": not unknown}
    if issues: response.warnings.extend(issues)
    return response
