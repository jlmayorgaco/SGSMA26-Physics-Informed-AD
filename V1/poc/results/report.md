# SGSMA 2026 POC Ablation Report

Run mode: full ablation.

## Winner

Best compactness-adjusted row: **E1_LSE + C1_RULES** with real Macro-F1=0.661, synthetic Macro-F1=0.333, real accuracy=0.667, synthetic accuracy=0.483, params=0, score=0.661.

## Gap Analysis

The requested synthetic held-out target is 0.99+ accuracy. The best row reached 0.483 synthetic accuracy and 0.667 accuracy on the nine real competition events.
8 of 16 combinations have synthetic-real Macro-F1 gap > 0.15. Treat those rows as distribution-shift red flags.

## Parameter Efficiency

The score uses Macro-F1_real - 0.03*log10(params). Zero-parameter rules and LSE remain important baselines because every learned component must earn its parameter penalty.

## Failure Cases

Classes with F1 below 0.5 for the winner: 0, 4, 7, 8.

## Lessons Learned

The POC keeps detector and localizer shared, so classifier differences mostly reflect residual representation quality rather than different event alignment. Large synthetic-real gaps mean the synthetic generator still misses real PMU quirks and stacked cyber-physical timing.
