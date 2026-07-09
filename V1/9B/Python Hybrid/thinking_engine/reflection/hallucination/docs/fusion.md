# Fusion Methods Reference

## Available methods

| Method | Formula | When to use |
|--------|---------|-------------|
| noisy_or | 1 - ∏(1 - wᵢpᵢ) | DEFAULT: one strong signal should dominate |
| weighted | Σ(wᵢpᵢ) / Σwᵢ | Interpretable, good when signals are equally reliable |
| bayesian | Sequential P(H) updates | When signals are conditionally independent evidence |
| dempster_shafer | Combination of mass functions | When verifiers can genuinely abstain (low confidence) |
| logistic | sigmoid(Σwᵢ·logit(pᵢ) / Σwᵢ) | Avoids extreme-probability compression of weighted avg |
| ensemble | Average of noisy_or + weighted + logistic | Robustness across distribution shapes |

## Signal format

`signals = [(verifier_name: str, failure_probability: float, weight: float), ...]`

- `failure_probability = (1 - score) * confidence + 0.5 * (1 - confidence)` — blends verifier's
  opinion with the "I don't know" neutral 0.5 in proportion to its confidence
- `weight = confidence * correlation_discount` — from CorrelationPipeline.to_fusion_signals()
