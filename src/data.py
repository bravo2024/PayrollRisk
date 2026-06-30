# Payroll advance risk data: wage, pay cycle, tenure, overdraft history
import numpy as np
import pandas as pd

FEATURES = ["wage_amount", "pay_frequency", "tenure_days", "overdraft_freq", "advance_ratio"]

def make_synthetic(n=3000, seed=42):
    rng = np.random.default_rng(seed)
    wage = rng.lognormal(7.5, 0.5, n).clip(100, 5000).round(2)
    freq = rng.choice([7, 14, 30], n, p=[0.2, 0.5, 0.3])
    tenure = rng.exponential(180, n).clip(1, 720).astype(int)
    overdraft = rng.poisson(0.3, n).clip(0, 5)
    advance = rng.beta(3, 5, n).round(3)
    df = pd.DataFrame({
        "wage_amount": wage, "pay_frequency": freq,
        "tenure_days": tenure, "overdraft_freq": overdraft,
        "advance_ratio": advance,
    })
    # Log-odds with known risk factors
    lo = (
        -1.5
        + 0.5 * np.clip(wage / 5000, 0, 1)
        - 0.3 * np.clip(freq / 30, 0, 1)
        - 0.6 * np.clip(tenure / 720, 0, 1)
        + 0.8 * np.clip(overdraft / 5, 0, 1)
        + 1.2 * advance
        + rng.normal(0, 0.4, n)
    )
    prob = 1 / (1 + np.exp(-lo))
    y = (prob > np.percentile(prob, 78)).astype(float)
    return {
        "X": df, "y": y, "features": FEATURES,
        "categorical_features": [], "numerical_features": FEATURES,
        "n_samples": n, "positive_rate": float(y.mean()),
    }