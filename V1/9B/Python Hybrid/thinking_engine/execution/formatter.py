# Copyright (C) 2026  Draco Studio and DUCNGUYEN-creator — GPL v3
"""
ResponseFormatter
==================
Post-processes a ResponseBuilder output dict into the final display
string — adds hallucination risk warnings, uncertainty tags, and
rewrite instructions if the HallucinationReport and Critic flagged
issues. Mirrors the end of engine_v1.py's process() return value
construction: thought_plan + calibrated_confidence + ethical_warning.
"""

from __future__ import annotations

from typing import Any, Dict, Optional


class ResponseFormatter:
    def format(
        self,
        response_dict: Dict[str, Any],
        hallucination_report: Optional[Dict[str, Any]] = None,
        ethical_warning: str = "",
        calibrated_confidence: float = 0.5,
        rewrite_instruction: str = "",
    ) -> str:
        text = response_dict.get("text", "").strip()

        if rewrite_instruction:
            text = f"{rewrite_instruction}\n\n{text}"

        if ethical_warning:
            text = f"{ethical_warning}\n\n{text}"

        if (
            hallucination_report
            and hallucination_report.get("risk_level") in ("high", "critical")
        ):
            risk_note = (
                f"\n\n[⚠️ HALLUCINATION RISK: {hallucination_report['risk_level'].upper()}] "
                f"score={hallucination_report.get('risk_score', 0):.2f} — "
                f"parts of this answer may be unverified."
            )
            text += risk_note

        return text
