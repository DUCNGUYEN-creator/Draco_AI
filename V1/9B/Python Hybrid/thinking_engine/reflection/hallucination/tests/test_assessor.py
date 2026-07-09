# Copyright (C) 2026  Draco Studio and DUCNGUYEN-creator — GPL v3
"""Smoke test: Assessor.assess() produces a valid HallucinationReport."""

from thinking_engine.reflection.hallucination import Assessor
from thinking_engine.reflection.hallucination.models.enums import RiskLevel


def test_assessor_wrong_arithmetic():
    a = Assessor()
    report = a.assess(
        "2 + 2 = 5.",
        context={"tool_results": [{"tool": "calculator", "output": "4", "ok": True}]},
        strategy_name="balanced",
    )
    assert report.risk_score >= 0.0
    assert report.risk_level in list(RiskLevel)
    assert report.n_verifiers_run == 6
    assert report.n_claims_checked >= 1
    # "4" vs "5" — ToolVerifier and NumericalVerifier should flag this
    assert report.risk_score > 0.3, f"Expected high risk for wrong arithmetic, got {report.risk_score}"


def test_assessor_correct_claim():
    a = Assessor()
    report = a.assess(
        "The capital of France is Paris.",
        context={"reasoning_path": ["France", "Capital", "Paris"]},
        strategy_name="fast",
    )
    assert report.n_verifiers_run == 2
    assert 0.0 <= report.risk_score <= 1.0


def test_assessor_empty_answer():
    a = Assessor()
    report = a.assess("", context={})
    assert report.risk_score == 0.0
    assert report.risk_level == RiskLevel.NONE


def test_assessor_telemetry():
    a = Assessor()
    a.assess("Hello world.", context={})
    snap = a.telemetry_snapshot()
    assert snap["request_count"] >= 1


if __name__ == "__main__":
    test_assessor_wrong_arithmetic()
    test_assessor_correct_claim()
    test_assessor_empty_answer()
    test_assessor_telemetry()
    print("ALL test_assessor TESTS PASSED")
