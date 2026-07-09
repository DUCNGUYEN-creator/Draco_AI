# Copyright (C) 2026  Draco Studio and DUCNGUYEN-creator — GPL v3
"""Smoke tests for all 9 verifiers."""

from thinking_engine.reflection.hallucination import verifiers, models


def test_numerical_verifier():
    nv = verifiers.NumericalVerifier()
    assert nv.verify("2 + 2 = 4", None, {})["score"] == 1.0
    assert nv.verify("2 + 2 = 5", None, {})["score"] == 0.0


def test_retrieval_verifier_no_evidence():
    rv = verifiers.RetrievalVerifier()
    r = rv.verify("Some claim with no evidence", models.EvidenceBundle(claim="x", items=[]), {})
    # No evidence => ABSTAIN (neutral score, very low confidence), not a
    # confident failure signal — see retrieval.py's module docstring for
    # the rationale (avoids false-positive CRITICAL risk on claims that
    # are correct but simply weren't backed by retrieved evidence).
    assert r["score"] == 0.5
    assert r["confidence"] < 0.1


def test_symbolic_verifier():
    sv = verifiers.SymbolicVerifier()
    assert sv.verify("A or not A is always true", None, {})["score"] == 1.0
    assert sv.verify("A and not A is always true", None, {})["score"] < 0.5


def test_tool_verifier_match():
    tv = verifiers.ToolVerifier()
    r = tv.verify("Kết quả là 42", None, {"tool_results": [{"tool": "calc", "output": "42", "ok": True}]})
    assert r["score"] == 1.0


def test_citation_verifier():
    cv = verifiers.CitationVerifier()
    r_valid = cv.verify("Theo [abc123def456], đúng.", None, {"known_citation_ids": ["abc123def456"]})
    r_fake  = cv.verify("Theo [aaa111bbb222], đúng.", None, {"known_citation_ids": ["abc123def456"]})
    assert r_valid["score"] == 1.0
    assert r_fake["score"] == 0.0


def test_ensemble():
    ens = verifiers.VerifierEnsemble([verifiers.NumericalVerifier(), verifiers.SymbolicVerifier()])
    results = ens.run_all("2 + 2 = 5", None, {})
    assert len(results) == 2


if __name__ == "__main__":
    test_numerical_verifier()
    test_retrieval_verifier_no_evidence()
    test_symbolic_verifier()
    test_tool_verifier_match()
    test_citation_verifier()
    test_ensemble()
    print("ALL test_verifiers TESTS PASSED")
