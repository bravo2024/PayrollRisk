# PayrollRisk

> Canadian payroll compliance risk scoring with CRA audit simulation.

Trains four classifiers on synthetic employer payroll data to flag T4 compliance risk. Dashboard provides data exploration, multi-model comparison, risk factor importance analysis (filing delays, remittance patterns, payroll frequency), and a Benford's Law audit simulation for detecting financial data fabrication.

## Quickstart

```bash
pip install -r requirements.txt
python train.py
pytest -q
streamlit run app.py
```

## Model Performance

Best model (Gradient Boosting) holdout results:

| Metric | Value |
|---|---|
| ROC AUC | 0.701 |
| Gini | 0.402 |
| KS Statistic | 0.327 |
| F1 Score | 0.324 |
| Accuracy | 0.726 |

5-fold CV AUC: 0.690 ± 0.030. Four models compared.

## Features

| Tab | What it does |
|---|---|
| **Explorer** | Employer dataset overview, compliance risk rate, distribution plots |
| **Model Lab** | Multi-model comparison, ROC/calibration curves, CV results |
| **Risk Factors** | CRA compliance driver importance, risk by filing delay and frequency |
| **Audit** | Benford's Law leading-digit analysis for financial data integrity |

## Repo Structure

```
PayrollRisk/
  src/         data, model, evaluate, persist, visualizations modules
  train.py     training pipeline (multi-model + CV)
  app.py       Streamlit dashboard
  tests/       pytest smoke test
  models/      saved model + metrics (gitignored)
```

## Data

Synthetic Canadian employer dataset: filing delay days, remittance frequency, payroll frequency, employee count, industry sector, T4 filing history, and compliance audit flag.

## License

MIT
