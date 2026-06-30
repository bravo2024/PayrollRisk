# Payroll risk metrics: roll rates and cash flow risk
import numpy as np

def roll_rate(repayment_hist, window=7):
    r = np.asarray(repayment_hist, dtype=float)
    rates = []
    for i in range(len(r)):
        start = max(0, i - window)
        rates.append(float(r[start:i+1].mean()))
    return rates

def cash_flow_risk(balance, income, expenses):
    net = np.asarray(income, dtype=float) - np.asarray(expenses, dtype=float)
    buffer = np.minimum(np.asarray(balance, dtype=float), net)
    return float(1 - buffer.sum() / max(net.sum(), 1))