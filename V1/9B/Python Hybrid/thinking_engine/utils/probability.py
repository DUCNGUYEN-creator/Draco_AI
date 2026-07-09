# Copyright (C) 2026  Draco Studio and DUCNGUYEN-creator — GPL v3
"""Probability-theory helpers reused by hallucination/fusion/bayesian.py
and reflection/confidence.py: noisy-OR combination, Bayesian updates."""

from __future__ import annotations

from typing import Iterable

from .math import clamp


def noisy_or(probs: Iterable[float]) -> float:
    """Combine independent evidence-of-failure probabilities:
    P(at least one true) = 1 - prod(1 - p_i). Standard for combining
    several weak hallucination signals into one risk estimate."""
    prod = 1.0
    for p in probs:
        prod *= (1.0 - clamp(p, 0.0, 1.0))
    return 1.0 - prod


def bayes_update(prior: float, likelihood_true: float, likelihood_false: float) -> float:
    """Single-step Bayesian update:
        posterior = P(E|H) P(H) / [P(E|H) P(H) + P(E|~H)(1-P(H))]
    """
    prior = clamp(prior, 1e-6, 1.0 - 1e-6)
    numer = likelihood_true * prior
    denom = numer + likelihood_false * (1.0 - prior)
    if denom <= 0:
        return prior
    return clamp(numer / denom, 0.0, 1.0)
