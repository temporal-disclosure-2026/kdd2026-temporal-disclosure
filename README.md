# kdd2026-temporal-disclosure

**Temporal Disclosure Dynamics in Child Counseling**

Anonymous submission to KDD 2026

## Overview

Code and data for reproducing survival analysis and
turn-budget optimization results.

**Dataset:** 3,236 Korean child counseling sessions
(ages 7–13, 2021–2023)

## Files

- `analysis_pipeline.py` — Survival and hazard analysis
- `semantic_validation.py` — Semantic anchoring validation
- `supervised_baselines.py` — Baseline comparison
- `session_metadata.csv` — Session-level metadata (N=3,236)
- `requirements.txt` — Package dependencies

## Keyword Lexicon

The 34-keyword risk detection lexicon is fully documented
in Appendix A (Table 5) of the paper, organized into
eight semantic categories. A separate lexicon file is
not required at runtime.

## Quick Start
```bash
pip install -r requirements.txt
```

### Run full analysis pipeline
```bash
python analysis_pipeline.py \
  --data_dir . \
  --output_dir ./output \
  --skip_lexicon_check
```

Outputs saved to `./output/`:
- `table_s1_dataset_characteristics.xlsx`
- `table_s2_msw_estimates.xlsx`
- `table_s3_turn_budget_sensitivity.xlsx`

### Run semantic validation
```bash
python semantic_validation.py \
  --data_dir . \
  --output_dir ./output
```

### Run supervised baselines
```bash
python supervised_baselines.py \
  --data_dir . \
  --output_dir ./output
```

## Requirements

- Python 3.9+
- lifelines 0.28.0
- statsmodels 0.14.0
- scikit-learn 1.3.0
- sentence-transformers 2.2.2

## License

MIT License (code) | Data for research use only

## Citation

Paper under review at KDD 2026.
