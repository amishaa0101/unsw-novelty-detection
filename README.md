# Detecting Unseen Cyberattacks with Statistical Novelty Detection (UNSW-NB15)

CS 328 Data Science course project, IIT Gandhinagar. Amisha Vaghela, Vaishnavi Chakole.

This README is a work in progress. The full version (pipeline, course topics, results, limitations) is generated at the end of the project. See `PROJECT_SPEC.md` for the design.

## Setup (Windows, Python 3.12)

```
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
```

Put the two official files in `data/raw/`:
`UNSW_NB15_training-set.csv` and `UNSW_NB15_testing-set.csv`.

## Commands

Run from the repo root with the venv Python (`.venv\Scripts\python`).

| Task | Command | Status |
|---|---|---|
| audit | `python scripts/run_audit.py` | Phase 1 |
| test | `python -m pytest tests` | later phases |
| quick | `python scripts/run_experiments.py --config configs/quick.yaml` | later phases |
| experiments | `python scripts/run_experiments.py --config configs/default.yaml` | later phases |
| figures | `python scripts/make_figures.py` | later phases |
