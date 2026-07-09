# Calibration Methods Reference

## Available methods

| Method | Params | Min samples | Best for |
|--------|--------|-------------|---------|
| platt | a, b (sigmoid) | 5 | General purpose, smooth curves |
| isotonic | step function (PAVA) | 10 | Non-sigmoid miscalibration |
| beta | α/β per decile | 5 | Sparse regions, Bayesian prior |
| temperature | T (logit scaling) | 3 | Very low data, fast convergence |
| histogram | bin counts | 5 | Baseline reference, no smoothing |
| ensemble | average of platt+temperature+beta | 5 | Robustness across distribution shapes |

## Two-point calibration design

1. **Per-verifier calibration** (CalibrationPipeline.calibrate_batch): each verifier's scores calibrated independently — corrects verifier-specific over/under-confidence
2. **Post-fusion calibration** (AssessmentPipeline): fused score calibrated again — corrects fusion-method bias (noisy_or systematically pushes scores up)
