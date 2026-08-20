from oil_analyst.models import AnalystResponse
from oil_analyst.validation import validate_response


def test_unreferenced_number_is_flagged():
    result = validate_response(AnalystResponse(answer="Demand rose by 3%."), [])
    assert not result.validation["valid"]


def test_nonempty_qualitative_answer_is_valid():
    assert validate_response(AnalystResponse(answer="Data are insufficient."), []).validation["valid"]
