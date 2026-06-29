from __future__ import annotations
import numpy as np; import pandas as pd
FEATURE_NAMES = ["employee_count","total_payroll_amount","overtime_pct","contractor_pct","industry_risk_score","location_risk_score","filing_delay_days","prior_audit_flags","payroll_frequency","avg_tenure_months"]
CATEGORICAL_FEATURES = ["payroll_frequency"]
NUMERICAL_FEATURES = ["employee_count","total_payroll_amount","overtime_pct","contractor_pct","industry_risk_score","location_risk_score","filing_delay_days","prior_audit_flags","avg_tenure_months"]
TARGET_NAME = "compliance_risk"
def make_synthetic(n=10000,seed=42):
    rng=np.random.default_rng(seed)
    df=pd.DataFrame({
        "employee_count": rng.poisson(lam=200,size=n).clip(1,5000),
        "total_payroll_amount": rng.lognormal(mean=13,sigma=1.5,size=n).clip(10000,100_000_000).astype(int),
        "overtime_pct": rng.beta(2,5,size=n).round(3),
        "contractor_pct": rng.beta(1.5,4,size=n).round(3),
        "industry_risk_score": rng.uniform(1,10,size=n).round(2),
        "location_risk_score": rng.uniform(1,10,size=n).round(2),
        "filing_delay_days": rng.exponential(scale=5,size=n).clip(0,60).round(1),
        "prior_audit_flags": rng.poisson(lam=0.3,size=n).clip(0,5),
        "payroll_frequency": rng.choice(["weekly","biweekly","monthly"],size=n,p=[0.25,0.50,0.25]),
        "avg_tenure_months": rng.exponential(scale=36,size=n).clip(1,240).astype(int),
    })
    emp=np.clip(df["employee_count"]/5000,0,1); pay=np.log(df["total_payroll_amount"]+1)/20
    over=df["overtime_pct"]; cont=df["contractor_pct"]; ind=df["industry_risk_score"]/10; loc=df["location_risk_score"]/10
    delay=np.clip(df["filing_delay_days"]/60,0,1); audit=np.clip(df["prior_audit_flags"]/5,0,1)
    freq_map={"weekly":0,"biweekly":0.5,"monthly":1}; freq=df["payroll_frequency"].map(freq_map).values
    ten=np.clip(df["avg_tenure_months"]/240,0,1)
    log_odds = -3.0 + 0.2*emp + 0.1*pay + 0.5*over + 0.4*cont + 0.4*ind + 0.3*loc + 0.6*delay + 0.7*audit + 0.2*freq - 0.2*ten + rng.normal(0,0.5,size=n)
    prob=1/(1+np.exp(-log_odds)); y=(prob>np.percentile(prob,88)).astype(np.float64)
    return {"X":df,"y":y,"features":FEATURE_NAMES,"df":df.assign(compliance_risk=y),"categorical_features":CATEGORICAL_FEATURES,"numerical_features":NUMERICAL_FEATURES,"n_samples":n,"n_features":len(FEATURE_NAMES),"positive_rate":y.mean()}
