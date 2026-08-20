"""Run the six documented MVP scenarios and save their raw results for review."""

from __future__ import annotations

import json
from pathlib import Path
from time import perf_counter
from typing import Any

from oil_analyst.factory import build_agent


PROJECT_ROOT = Path(__file__).resolve().parents[1]
QUESTIONS_PATH = PROJECT_ROOT / "eval" / "demo_questions.json"
RESULTS_PATH = PROJECT_ROOT / "eval" / "demo_results.md"


def _json_block(value: Any) -> str:
    return "```json\n" + json.dumps(value, ensure_ascii=False, indent=2) + "\n```"


def main() -> int:
    questions = json.loads(QUESTIONS_PATH.read_text(encoding="utf-8"))
    agent = build_agent()
    sections = [
        "# Demo check results",
        "",
        "Raw smoke-check output from the existing production pipeline. No automatic PASS/FAIL evaluation is applied.",
        "",
    ]

    total = len(questions)
    for index, item in enumerate(questions, 1):
        scenario, query = item["scenario"], item["query"]
        print(f"[{index}/{total}] {scenario}", flush=True)
        started = perf_counter()
        try:
            response = agent.invoke(query)
            elapsed = perf_counter() - started
            sources = [source.model_dump(mode="json") for source in response.sources]
            filters = response.metadata_filters.model_dump(exclude_none=True)
            forecast = response.forecast.model_dump(mode="json") if response.forecast else None
            sections.extend([
                f"## {index}. {scenario}",
                "",
                "**Query**",
                "",
                query,
                "",
                f"**Execution time:** {elapsed:.2f} seconds",
                "",
                "**Route**",
                "",
                _json_block(response.route),
                "",
                "**Answer**",
                "",
                response.answer,
                "",
                "**Uncertainty**",
                "",
                response.uncertainty or "None",
                "",
                "**Sources**",
                "",
                _json_block(sources),
                "",
                "**Metadata filters**",
                "",
                _json_block(filters),
                "",
                "**Forecast result**",
                "",
                _json_block(forecast),
                "",
            ])
        except Exception as exc:
            elapsed = perf_counter() - started
            sections.extend([
                f"## {index}. {scenario}",
                "",
                "**Query**",
                "",
                query,
                "",
                f"**Execution time:** {elapsed:.2f} seconds",
                "",
                "**Execution error**",
                "",
                f"`{type(exc).__name__}: {exc}`",
                "",
            ])

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text("\n".join(sections), encoding="utf-8")
    print(f"Results saved to {RESULTS_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
