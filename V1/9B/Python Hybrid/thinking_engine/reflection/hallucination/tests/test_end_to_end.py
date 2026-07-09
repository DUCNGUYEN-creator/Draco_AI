# Copyright (C) 2026  Draco Studio and DUCNGUYEN-creator — GPL v3
"""End-to-end smoke test: full 6-stage pipeline via AssessmentPipeline."""

from thinking_engine.reflection.hallucination.pipeline.assessment_pipeline import AssessmentPipeline
from thinking_engine.reflection.hallucination.models.enums import RiskLevel


def test_pipeline_end_to_end():
    pipeline = AssessmentPipeline()
    report = pipeline.assess(
        answer="2 + 2 = 5. Python was invented in 1990.",
        context={"tool_results": [{"tool": "calculator", "output": "4", "ok": True}]},
        strategy_name="balanced",
    )
    assert report.n_claims_checked >= 1
    assert report.n_verifiers_run == 6
    assert 0.0 <= report.risk_score <= 1.0
    assert report.risk_level in list(RiskLevel)
    report_dict = report.as_dict()
    assert "risk_score" in report_dict
    assert "top_issues" in report_dict
    assert "per_claim" in report_dict


def test_pipeline_fast_strategy():
    pipeline = AssessmentPipeline()
    report = pipeline.assess(
        "The sky is blue.",
        context={},
        strategy_name="fast",
    )
    assert report.n_verifiers_run == 2


def test_pipeline_paranoid_strategy():
    pipeline = AssessmentPipeline()
    report = pipeline.assess(
        "2 + 2 = 4.",
        context={},
        strategy_name="paranoid",
    )
    assert report.n_verifiers_run == 9


if __name__ == "__main__":
    test_pipeline_end_to_end()
    test_pipeline_fast_strategy()
    test_pipeline_paranoid_strategy()
    print("ALL test_end_to_end TESTS PASSED")
