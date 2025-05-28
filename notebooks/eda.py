#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Risk Score v2 – Istanbul district-level multi-factor risk model
==============================================================
Implements the improved pipeline discussed on 2025-05-27:
    • Robust + Quantile (uniform) scaling
    • Direction correction for protective features
    • Four-pillar index construction  (Hazard, Exposure, Vulnerability, Capacity)
    • Initial weights via entropy method   → fine-tuned by Optuna (L2-regularised, Σw=1)
    • Log-multiplicative risk formula (exp-sum form)
    • Natural breaks with k-means (k=5) for Very-Low → Very-High classes
    • CSV/Excel output
Run:
    python risk_score_v2.py --csv_in data.csv --csv_out risk_scores.xlsx --optuna 50
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Tuple, Dict
import argparse, warnings
import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler, QuantileTransformer
from sklearn.cluster import KMeans
import optuna

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# ------------------------------ CONFIG ------------------------------------
# mapping of column groups to the four risk pillars
HAZARD_COLS: List[str] = [
    "Deprem Puan",          # 0-1 normalized (higher = more seismic hazard)
    "Bölge Riski"           # composite local hazard score
]
EXPOSURE_COLS: List[str] = [
    "Toplam Nüfus", "bina_yogunluk", "ekonomik_deger_endeks"
]
VULN_COLS: List[str] = [
    "Aylık Ortalama Hane Geliri", "tasarruf_oran", "çocuk_oran",
    "yaslı_oran", "işsizlik_oran", "okumamış_oran"
]
CAPACITY_COLS: List[str] = [
    "Sağlık Tesisi / 10k", "İtfaiye / km2", "yol_erişim_endeks"
]

# features whose artışı *azaltıcı* etki yapar (koruyucu) → 1-x dönüşümü
NEGATIVE_DIRS = set([
    *CAPACITY_COLS,                  # kapasite sütunları risk düşürür
    "Aylık Ortalama Hane Geliri",  # yüksek gelir koruyucu (VULN)
    "Okuryazar Oran"                # örnek koruyucu değişken
])

SEED = 42
EPS  = 1e-6                            # to avoid log(0)

# -------------------------------------------------------------------------
# 1.  DATA I/O & BASIC PREPROCESSING
# -------------------------------------------------------------------------

def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    # Lokasyon birleşik anahtar
    df["Lokasyon"] = (
        df["İl Adı"].astype(str) + "_" +
        df["İlçe Adı"].astype(str) + "_" +
        df["Mahalle Adı"].astype(str)
    )
    df = df.set_index("Lokasyon").drop(columns=["İl Adı", "İlçe Adı", "Mahalle Adı"], errors="ignore")
    # Temel filtre – çok küçük nüfuslu mahalleler hariç
    if "Toplam Nüfus" in df.columns:
        df = df[df["Toplam Nüfus"] >= 200]
    return df

# -------------------------------------------------------------------------
# 2.  SCALING & DIRECTION ADJUSTMENT
# -------------------------------------------------------------------------

def robust_quantile_scale(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    """RobustScaler → QuantileTransformer(uniform) cascaded"""
    scaler = RobustScaler().fit(df[cols])
    arr    = scaler.transform(df[cols])
    qtf    = QuantileTransformer(output_distribution="uniform", random_state=SEED)
    arr_q  = qtf.fit_transform(arr)
    df[cols] = arr_q
    # ensure 0-1 clipping
    df[cols] = df[cols].clip(0, 1)
    return df

def apply_directions(df: pd.DataFrame) -> pd.DataFrame:
    df2 = df.copy()
    for col in (set(df.columns) & NEGATIVE_DIRS):
        df2[col] = 1.0 - df2[col]
    return df2

# -------------------------------------------------------------------------
# 3.  PILLAR INDICES
# -------------------------------------------------------------------------

def mean_or_zero(df: pd.DataFrame, cols: List[str]) -> pd.Series:
    common = [c for c in cols if c in df.columns]
    if not common:
        raise ValueError(f"None of the required columns found for: {cols}")
    return df[common].mean(axis=1)

def build_indices(df: pd.DataFrame) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    H = mean_or_zero(df, HAZARD_COLS)
    E = mean_or_zero(df, EXPOSURE_COLS)
    V = mean_or_zero(df, VULN_COLS)
    C = mean_or_zero(df, CAPACITY_COLS)  # protective, zaten 1-x uygulanmış durumda
    return H+EPS, E+EPS, V+EPS, C+EPS

# -------------------------------------------------------------------------
# 4.  ENTROPY WEIGHTS  (objective, unsupervised init)
# -------------------------------------------------------------------------

def entropy_weights(df_idx: pd.DataFrame) -> Dict[str, float]:
    # df_idx columns = [H,E,V,C] scaled 0-1
    P = df_idx / df_idx.sum(axis=0)           # probabilistic share
    E = - (P * np.log(P + EPS)).sum(axis=0)   # Shannon entropy per feature
    d = 1 - E/np.log(len(df_idx))             # information utility
    w = d / d.sum()
    return dict(zip(df_idx.columns, w))

# -------------------------------------------------------------------------
# 5.  OPTUNA FINE-TUNE  (Σw=1, L2 penalty)
# -------------------------------------------------------------------------

def optimise_weights(df_idx: pd.DataFrame, base_w: Dict[str,float], n_trials: int=50, l2_lambda: float=0.1) -> Dict[str,float]:
    columns = list(df_idx.columns)
    def objective(trial):
        raw = [trial.suggest_float(c, 0.05, 1.0) for c in columns]
        w    = np.array(raw) / np.sum(raw)          # Σw=1 constraint
        l2   = l2_lambda * np.sum((w - np.array([base_w[c] for c in columns]))**2)
        # Compute risk & its variance (want wider spread)  ↑variance ↓penalty
        log_risk = (df_idx * w[[0,1,2]]).sum(axis=1) - w[3] * np.log(df_idx['C'])  # not used (placeholder)
        # to simplify: maximise std of risk (spread) – penalty
        rs = calc_risk(df_idx, dict(zip(columns, w)))
        score = np.std(rs) - l2
        return score
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=SEED))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    best_raw = study.best_params
    w_arr = np.array([best_raw[c] for c in columns])
    w_arr = w_arr / w_arr.sum()
    return dict(zip(columns, w_arr))

# -------------------------------------------------------------------------
# 6.  RISK SCORE (log multiplicative)
# -------------------------------------------------------------------------

def calc_risk(df_idx: pd.DataFrame, w: Dict[str,float]) -> pd.Series:
    """RS = 100 * exp( wH ln H + wE ln E + wV ln V − wC ln C )"""
    ln_part = (
        w['H'] * np.log(df_idx['H']) +
        w['E'] * np.log(df_idx['E']) +
        w['V'] * np.log(df_idx['V']) -
        w['C'] * np.log(df_idx['C'])
    )
    return 100.0 * np.exp(ln_part)

# -------------------------------------------------------------------------
# 7.  CLASSIFICATION (natural breaks – k-means)
# -------------------------------------------------------------------------

def classify_risk(rs: pd.Series, k: int = 5) -> pd.Series:
    km = KMeans(n_clusters=k, random_state=SEED).fit(rs.values.reshape(-1,1))
    centers = km.cluster_centers_.flatten()
    order   = centers.argsort()               # small→large risk
    mapping = {old:new for new,old in enumerate(order)}
    labels  = pd.Series(km.labels_, index=rs.index).map(mapping)
    classes = {0:'Very-Low',1:'Low',2:'Medium',3:'High',4:'Very-High'}
    return labels.map(classes)

# -------------------------------------------------------------------------
# 8.  PIPELINE
# -------------------------------------------------------------------------

def pipeline(csv_in: str, csv_out: str, n_trials: int):
    df = load_data(csv_in)

    # All candidate numeric features (scale everything then slice)
    num_cols = df.select_dtypes(include=['int64','float64']).columns.tolist()
    df = robust_quantile_scale(df, num_cols)
    df = apply_directions(df)

    # Build pillar indices
    H,E,V,C = build_indices(df)
    df_idx  = pd.DataFrame({'H':H,'E':E,'V':V,'C':C})

    # Entropy base weights → Optuna fine-tune
    base_w  = entropy_weights(df_idx)
    w_opt   = optimise_weights(df_idx, base_w, n_trials=n_trials)

    # Risk score + class
    rs = calc_risk(df_idx, w_opt)
    df['RiskScore'] = rs
    df['RiskClass'] = classify_risk(rs)

    # Export
    out_path = Path(csv_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df[['RiskScore','RiskClass']].to_excel(out_path)
    print("✔  Results saved →", out_path)
    print("Optimum weights:", w_opt)
    print(df['RiskClass'].value_counts().sort_index())

# -------------------------------------------------------------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv_in",  default="data.csv")
    ap.add_argument("--csv_out", default="risk_scores.xlsx")
    ap.add_argument("--optuna",  type=int, default=50, help="Optuna trial count (default=50)")
    args = ap.parse_args()
    pipeline(args.csv_in, args.csv_out, args
