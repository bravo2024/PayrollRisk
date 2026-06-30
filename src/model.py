# Payroll advance risk model: logistic regression with standardized features
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

def fit_and_evaluate(data, seed=42):
    X = data["X"].values.astype(float)
    y = np.asarray(data["y"])
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, stratify=y, random_state=seed)
    scaler = StandardScaler()
    Xtr = scaler.fit_transform(Xtr)
    Xte = scaler.transform(Xte)
    model = LogisticRegression(class_weight="balanced", random_state=seed)
    model.fit(Xtr, ytr)
    proba = model.predict_proba(Xte)[:, 1]
    from sklearn.metrics import roc_auc_score
    auc = float(roc_auc_score(yte, proba))
    return (
        {"model": model, "scaler": scaler},
        {
            "auc": auc,
            "coefficients": [float(c) for c in model.coef_[0]],
        },
    )