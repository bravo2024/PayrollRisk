from __future__ import annotations
import sys; from pathlib import Path; sys.path.insert(0, str(Path(__file__).parent))
import numpy as np, pandas as pd, streamlit as st, matplotlib.pyplot as plt
from src.data import make_synthetic, TARGET_NAME
from src.model import train_all_models, cross_validate
from src.visualizations import *
st.set_page_config(page_title="PayrollRisk | Pinpoint Canadian Payroll Compliance", layout="wide", page_icon="\U0001f4b3")
with st.sidebar:
    st.header("\u2699 Config"); n=st.slider("Employers",2000,20000,10000,1000); tau=st.slider("Threshold",0.05,0.95,0.50,0.05)
    st.caption("Pinpoint | Canadian Payroll & T4 Compliance Risk")
data=make_synthetic(n=n); b=train_all_models(data)
y_test=b["y_test"]; y_probas={n:b["results"][n]["y_proba"] for n in b["results"]}
best=max(b["results"],key=lambda n: b["results"][n]["metrics"].get("roc_auc",0))
c1,c2,c3,c4=st.columns(4)
c1.metric("Samples",f"{n:,}"); c2.metric("Compliance Risk Rate",f"{data['positive_rate']:.1%}")
c3.metric("Best AUC",f"{b['results'][best]['metrics']['roc_auc']:.4f}"); c4.metric("Best",best)
t1,t2,t3,t4=st.tabs(["\U0001f4ca Explorer","\U0001f52c Model Lab","\U0001f3af Risk Factors","\U0001f4cb Audit"])
with t1:
    st.dataframe(data["df"].head(50),use_container_width=True,height=200)
    fig,ax=plt.subplots(figsize=(5,3)); _style()
    ax.bar(["Compliant","At Risk"],[1-data["positive_rate"],data["positive_rate"]],color=["#22c55e","#f43f5e"])
    for i,v in enumerate([1-data["positive_rate"],data["positive_rate"]]): ax.text(i,v+.01,f"{v:.1%}",ha="center",color="white")
    ax.set_title("Compliance Risk Distribution",color="white"); ax.grid(True,alpha=.2)
    st.pyplot(fig)
with t2:
    rows=[{**{"Model":n},**{k:f"{v:.4f}" for k,v in r["metrics"].items() if k!="confusion_matrix"}} for n,r in b["results"].items()]
    st.dataframe(pd.DataFrame(rows).set_index("Model"),use_container_width=True)
    col_a,col_b=st.columns(2)
    with col_a: st.pyplot(plot_roc_curve(y_test,y_probas))
    with col_b: st.pyplot(plot_calibration_curve(y_test,y_probas))
    st.pyplot(plot_confusion_matrix(y_test,b["results"]["XGBoost"]["y_pred"],"XGBoost"))
    cv=cross_validate(data); cvr=[{"Model":n,"AUC":f"{s['roc_auc']['mean']:.4f}","\u00b1":f"\u00b1{s['roc_auc']['std']:.4f}"} for n,s in cv.items()]
    st.dataframe(pd.DataFrame(cvr).set_index("Model"),use_container_width=True)
with t3:
    st.subheader("Risk Factor Analysis")
    st.latex(r"\text{logit}(p) = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + \cdots + \beta_k x_k")
    st.caption("Pinpoint's payroll compliance engine flags employers at risk of CRA audit based on filing delays, remittance patterns, and payroll frequency. Key CDN regulations: T4 filing due Feb 28, ROE within 5 days of interruption, CPP/EI remittance deadlines.")
    importances=b["models"]["XGBoost"].feature_importances_
    fi=pd.DataFrame({"feature":data["features"],"importance":importances}).sort_values("importance",ascending=True)
    fig,ax=plt.subplots(figsize=(6,6)); _style()
    ax.barh(fi["feature"],fi["importance"],color="#22d3ee"); ax.set_title("Risk Driver Importance",color="white")
    ax.set_xlabel("Importance"); ax.grid(True,alpha=.2)
    st.pyplot(fig)
    col_a,col_b=st.columns(2)
    with col_a:
        delay_bins=pd.cut(data["df"]["filing_delay_days"],bins=[0,1,5,15,60],labels=["None","Low","Medium","High"])
        risk_by_delay=data["df"].groupby(delay_bins,observed=True)[TARGET_NAME].mean()
        fig,ax=plt.subplots(figsize=(5,3)); _style()
        ax.bar(range(4),risk_by_delay.values,color=["#22c55e","#fbbf24","#f97316","#f43f5e"])
        ax.set_xticks(range(4)); ax.set_xticklabels(risk_by_delay.index); ax.set_title("Risk by Filing Delay",color="white")
        ax.grid(True,alpha=.2); st.pyplot(fig)
    with col_b:
        ind_risk=data["df"].groupby("payroll_frequency")[TARGET_NAME].mean()
        fig,ax=plt.subplots(figsize=(5,3)); _style()
        ax.bar(ind_risk.index,ind_risk.values,color=["#22d3ee","#f97316","#22c55e"])
        ax.set_title("Risk by Payroll Frequency",color="white"); ax.grid(True,alpha=.2); st.pyplot(fig)
with t4:
    st.subheader("Audit Simulation")
    st.latex(r"P(d) = \log_{10}\!\left(1 + \frac{1}{d}\right), \quad d \in \{1,2,\ldots,9\}")
    st.caption("Benford's Law: leading-digit distribution in natural financial data; deviations flag possible fraud or data fabrication.")
    st.latex(r"\chi^2 = \sum_{i=1}^k \frac{(O_i - E_i)^2}{E_i}")
    st.caption("Chi-square goodness-of-fit test compares observed vs expected digit frequencies under Benford's Law.")
    st.latex(r"\text{Audit Score} = \sum w_i \cdot \text{Risk}_i")
    st.latex(r"\text{Materiality} = \max\!\big(0.5\% \times \text{Revenue},\; 1\% \times \text{Assets}\big)")
    st.caption("Audit materiality threshold per ISA 320: misstatements below this level are considered immaterial for financial statement users.")
    st.latex(r"\text{MUS Sample Size} = \frac{\text{BV} \times \text{Confidence Factor}}{\text{Materiality} - (\text{Expected Error} \times \text{Expansion Factor})}")
    st.caption("Monetary Unit Sampling (MUS): dollar-unit selection weighting larger items; standard for substantive audit testing of account balances.")
    audit_score=b["results"]["XGBoost"]["y_proba"]
    high_risk=(audit_score>0.7).sum(); med_risk=((audit_score>0.4)&(audit_score<=0.7)).sum(); low_risk=(audit_score<=0.4).sum()
    c1,c2,c3=st.columns(3)
    c1.metric("High Risk",f"{high_risk:,}"); c2.metric("Medium Risk",f"{med_risk:,}"); c3.metric("Low Risk",f"{low_risk:,}")
    fig,ax=plt.subplots(figsize=(8,4)); _style()
    ax.hist(audit_score,bins=50,color="#22d3ee",alpha=0.6,edgecolor="#1a1f2e")
    ax.axvline(0.4,color="#fbbf24",ls="--",lw=2,label="Medium Threshold")
    ax.axvline(0.7,color="#f43f5e",ls="--",lw=2,label="High Threshold")
    ax.set_title("Payroll Risk Score Distribution — CRA Audit Likelihood",color="white"); ax.legend(); ax.grid(True,alpha=.2)
    st.pyplot(fig)
    st.caption("CRA payroll audit risk tiers: High risk (>0.7) warrants immediate T4/ROE remediation; Medium (0.4-0.7) triggers proactive filing review; Low (<0.4) indicates normal compliance. Pinpoint integrates with CDN payroll providers (Ceridian, ADP, Payworks) to automate compliance monitoring.")
