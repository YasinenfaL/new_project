#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Risk Score v3 – Istanbul district-level multi-factor risk model (MinMax scaling + new Infrastructure index)
==============================================================
Implements the pipeline with a fifth "Infrastructure" pillar using original service-related variables:
    • MinMaxScaler scaling
    • Direction correction for protective features
    • Five-pillar index construction (Hazard, Exposure, Vulnerability, Capacity, Infrastructure)
    • Initial weights via entropy method → fine-tuned by Optuna (L2-regularised, Σw=1)
    • Log-multiplicative risk formula (exp-sum form) with protective pillars negative
    • Natural breaks with k-means (k=5) for Very-Low → Very-High classes
    • Excel output
Run:
    python risk_score_v3.py --csv_in data.csv --csv_out risk_scores.xlsx --optuna 50
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Tuple, Dict
import argparse, warnings
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
import optuna

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# ------------------------------ CONFIG ------------------------------------
HAZARD_COLS: List[str] = [
    "Deprem Puan",
    "Bölge Riski"
]
EXPOSURE_COLS: List[str] = [
    "Toplam Nüfus",
    "Konut Yoğunluğu (Konut/Km2)",
    "İş Yeri Yoğunluğu (İş Yeri/Km2)",
    "Alan Başına Düşen İnsan Sayısı (ha)"
]
VULN_COLS: List[str] = [
    "Aylık Ortalama Hane Geliri",
    "Aylık Hane Tasarrufu",
    "0-15 Kişi Sayısı",
    "55+ Kişi Sayısı",
    "Okumamış Kişi Sayısı",
    "Ortalama Eğitim Süresi (Yıl)"
]
CAPACITY_COLS: List[str] = [
    "Konut Sayısı",
    "İş Yeri Sayısı",
    "Toplam Yerleşim Sayısı"
]
# New Infrastructure pillar using service access variables
INFRA_STRUCT_COLS: List[str] = [
    "Sağlık Tesisi / 10k",
    "İtfaiye / km2",
    "yol_erişim_endeks",
    "Okuryazar Oran"
]

# Protective features: increases reduce risk
NEGATIVE_DIRS = set([
    *CAPACITY_COLS,
    *INFRA_STRUCT_COLS,
    "Aylık Ortalama Hane Geliri",
    "Aylık Hane Tasarrufu"
])

SEED = 42
EPS  = 1e-6

# -------------------------------------------------------------------------
# 1. DATA I/O & PREPROCESSING
# -------------------------------------------------------------------------

def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["Lokasyon"] = (
        df["İl Adı"].astype(str) + "_" +
        df["İlçe Adı"].astype(str) + "_" +
        df["Mahalle Adı"].astype(str)
    )
    df = df.set_index("Lokasyon").drop(columns=["İl Adı", "İlçe Adı", "Mahalle Adı"], errors="ignore")
    if "Toplam Nüfus" in df.columns:
        df = df[df["Toplam Nüfus"] >= 200]
    return df

# -------------------------------------------------------------------------
# 2. SCALING & DIRECTION
# -------------------------------------------------------------------------

def minmax_scale(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    scaler = MinMaxScaler().fit(df[cols])
    df[cols] = scaler.transform(df[cols])
    return df

def apply_directions(df: pd.DataFrame) -> pd.DataFrame:
    df2 = df.copy()
    for col in set(df.columns) & NEGATIVE_DIRS:
        df2[col] = 1.0 - df2[col]
    return df2

# -------------------------------------------------------------------------
# 3. PILLAR INDICES
# -------------------------------------------------------------------------

def mean_or_zero(df: pd.DataFrame, cols: List[str]) -> pd.Series:
    common = [c for c in cols if c in df.columns]
    if not common:
        raise ValueError(f"No columns for: {cols}")
    return df[common].mean(axis=1)

def build_indices(df: pd.DataFrame) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series, pd.Series]:
    H = mean_or_zero(df, HAZARD_COLS) + EPS
    E = mean_or_zero(df, EXPOSURE_COLS) + EPS
    V = mean_or_zero(df, VULN_COLS) + EPS
    C = mean_or_zero(df, CAPACITY_COLS) + EPS
    I = mean_or_zero(df, INFRA_STRUCT_COLS) + EPS
    return H, E, V, C, I

# -------------------------------------------------------------------------
# 4. ENTROPY WEIGHTS
# -------------------------------------------------------------------------

def entropy_weights(df_idx: pd.DataFrame) -> Dict[str, float]:
    P = df_idx / df_idx.sum(axis=0)
    E = - (P * np.log(P + EPS)).sum(axis=0)
    d = 1 - E/np.log(len(df_idx))
    w = d / d.sum()
    return dict(zip(df_idx.columns, w))

# -------------------------------------------------------------------------
# 5. OPTUNA FINE-TUNE
# -------------------------------------------------------------------------

def optimise_weights(df_idx: pd.DataFrame, base_w: Dict[str,float], n_trials: int=50, l2_lambda: float=0.1) -> Dict[str,float]:
    cols = list(df_idx.columns)
    def obj(trial):
        raw = [trial.suggest_float(c, 0.05, 1.0) for c in cols]
        w = np.array(raw) / np.sum(raw)
        penalty = l2_lambda * np.sum((w - np.array([base_w[c] for c in cols]))**2)
        rs = calc_risk(df_idx, dict(zip(cols, w)))
        return np.std(rs) - penalty
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=SEED))
    study.optimize(obj, n_trials=n_trials, show_progress_bar=False)
    best = study.best_params
    arr = np.array([best[c] for c in cols]); arr /= arr.sum()
    return dict(zip(cols, arr))

# -------------------------------------------------------------------------
# 6. RISK SCORE (log-multiplicative)
# -------------------------------------------------------------------------

def calc_risk(df_idx: pd.DataFrame, w: Dict[str,float]) -> pd.Series:
    part = (
        w['H']*np.log(df_idx['H']) +
        w['E']*np.log(df_idx['E']) +
        w['V']*np.log(df_idx['V']) -
        w['C']*np.log(df_idx['C']) -
        w['I']*np.log(df_idx['I'])
    )
    return 100.0 * np.exp(part)

# -------------------------------------------------------------------------
# 7. CLASSIFICATION
# -------------------------------------------------------------------------

def classify_risk(rs: pd.Series, k: int = 5) -> pd.Series:
    km = KMeans(n_clusters=k, random_state=SEED).fit(rs.values.reshape(-1,1))
    order = km.cluster_centers_.flatten().argsort()
    mapping = {old:new for new,old in enumerate(order)}
    labels = pd.Series(km.labels_, index=rs.index).map(mapping)
    classes = {0:'Very-Low',1:'Low',2:'Medium',3:'High',4:'Very-High'}
    return labels.map(classes)

# -------------------------------------------------------------------------
# 8. PIPELINE
# -------------------------------------------------------------------------

def pipeline(csv_in: str, csv_out: str, n_trials: int):
    df = load_data(csv_in)
    num_cols = df.select_dtypes(include=['int64','float64']).columns.tolist()
    df = minmax_scale(df, num_cols)
    df = apply_directions(df)

    H,E,V,C,I = build_indices(df)
    df_idx = pd.DataFrame({'H':H,'E':E,'V':V,'C':C,'I':I})

    base_w = entropy_weights(df_idx)
    w_opt  = optimise_weights(df_idx, base_w, n_trials=n_trials)

    rs = calc_risk(df_idx, w_opt)
    df['RiskScore'] = rs
    df['RiskClass'] = classify_risk(rs)

    out = Path(csv_out); out.parent.mkdir(parents=True, exist_ok=True)
    df[['RiskScore','RiskClass']].to_excel(out)
    print("✔ Results saved →", out)
    print("Optimum weights:", w_opt)
    print(df['RiskClass'].value_counts().sort_index())

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv_in",  default="data.csv")
    ap.add_argument("--csv_out", default="risk_scores.xlsx")
    ap.add_argument("--optuna",  type=int, default=50, help="Optuna trials")
    args = ap.parse_args()
    pipeline(args.csv_in, args.csv_out, args.optuna)
