# Copyright (C) 2026  Draco Studio and DUCNGUYEN-creator — GPL v3
"""
ContradictionAnalyzer
========================
Aggregates ALL issues reported by ContradictionVerifier (and any other
verifier that reported contradiction-flavoured issues) across every
claim in a response, producing one consolidated list for the final
HallucinationReport.top_issues — avoids the report repeating near-
identical contradiction messages per-claim.
"""

from __future__ import annotations

from typing import Dict, List


class ContradictionAnalyzer:
    def consolidate(self, per_claim_results: List[List[dict]]) -> List[str]:
        seen: Dict[str, int] = {}
        for claim_results in per_claim_results:
            for r in claim_results:
                if r.get("verifier") != "contradiction":
                    continue
                for issue in r.get("issues", []):
                    seen[issue] = seen.get(issue, 0) + 1
        # Most-repeated contradiction patterns first — likely systemic, not one-off.
        ranked = sorted(seen.items(), key=lambda kv: kv[1], reverse=True)
        return [issue for issue, _ in ranked]
