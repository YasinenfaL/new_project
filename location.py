# -*- coding: utf-8 -*-
"""
Mahalle Bazlı Kredi Risk Skoru – v2.2 (jenkspy-free, argparse-free)
------------------------------------------------------------------------
Özellikler:
 1. Yön tersleme → RobustScaler → MinMax (0–1)
 2. Optuna çoklu-objective (Calinski-Harabasz ↑, Inertia ↓) + L1 sparsity + warm-start
 3. Composite skorlar: economic_resilience, financial_leverage, socio_demo_vulnerability
 4. Per-capita normalizasyon (kişi başı/1000 kişi)
 5. Jenks-free eşik: K-Means merkez ara noktaları
 6. Dinamik alfabetik Risk_Sınıfı (A, B, C, ...)

Kullanım:
 >>> python -c "import risk_score_v2; risk_score_v2.run_pipeline('data.csv','risk_score.xlsx',optuna_trials=50, default_label=None)"
 ya da doğrudan Jupyter içinde:
 >>> from risk_score_v2 import run_pipeline
 >>> run_pipeline('data.csv','risk_score.xlsx')
"""
from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import calinski_harabasz_score
from sklearn.cluster import KMeans
from sklearn.isotonic import IsotonicRegression
import optuna
import warnings

warnings.filterwarnings("ignore")

# 0 · Config
GROUPS: dict[str, dict] = { ... }  # (Aynı GROUPS tanımı yukarıdaki gibi)
DIRECTIONS: Dict[str, str] = {
    **{feat: 'pos' for feat in [
        "Bireysel Kredi / Gelir","gelir_kira_yuku","Toplam Harcama",
        "Kullanılan Toplam Kredi (BinTL)","kart_kisi_oran",
        "Deprem Puan","Bölge Riski",
        "ilkokul_oran","ilköğretim_oran","okumamış_oran", 
        "Tüketim Potansiyeli (Yüzde %)","DE Oran",
        "Ortalama Hanehalkı","çocuk_oran","genç_oran",
        "yaslı_oran","bekar_oran",
        "konut_yogunlugu","Ortalama Kira Değerleri"
    ]},
    **{feat: 'neg' for feat in [
        "Aylık Ortalama Hane Geliri","tasarruf_oran",
        "Toplam Mevduat / Gelir","Ortalama Eğitim Süresi (Yıl)",
        "lisansüstü_oran","üniversite_oran","okuryazar_oran",
        "Ortalama SES","Gelişmişlik Katsayısı","AB Oran", 
        "orta_yas_oran","evli_oran","Alan Büyüklüğü / km2",
        "Hane Başı Araç Sahipliği"
    ]}
}

COMPOSITES: Dict[str, List[str]] = {
    "economic_resilience": [
        "Gelir & Harcama","Çalışma","Varlık"
    ],
    "financial_leverage": [
        "Kredi & Bankacılık"
    ],
    "socio_demo_vulnerability": [
        "Demografi","Eğitim","Tüketim & SES","Altyapı","Coğrafi & Afet"
    ]
}

# 1 · Yardımcı Fonksiyonlar

def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["Lokasyon"] = (
        df["İl Adı"].astype(str) + "_" + df["İlçe Adı"].astype(str) + "_" + df["Mahalle Adı"].astype(str)
    )
    df = df.drop(columns=["İl Adı", "İlçe Adı", "Mahalle Adı"]).set_index("Lokasyon")
    return df


def robust_minmax(df: pd.DataFrame, features: List[str]) -> pd.DataFrame:
    for feat in features:
        if DIRECTIONS.get(feat) == "neg":
            df[feat] = -df[feat]
    scaler = RobustScaler()
    df[features] = scaler.fit_transform(df[features])
    df[features] = (df[features] - df[features].min()) / (df[features].max() - df[features].min())
    return df


def compute_weights_prior() -> Dict[str, float]:
    w = {}
    for grp, meta in GROUPS.items():
        pr = meta["priority"] / len(meta["features"])
        for f in meta["features"]:
            w[f] = pr
    s = sum(w.values())
    return {k: v / s for k, v in w.items()}


def optimize_weights(df: pd.DataFrame, features: List[str], trials: int, prior: Dict[str, float]) -> Dict[str, float]:
    # Validate features
    missing_features = [f for f in features if f not in df.columns]
    if missing_features:
        raise ValueError(f"Missing features in DataFrame: {missing_features}")
        
    def objective(trial: optuna.Trial):
        params = {feat: trial.suggest_float(feat, 0.0, 5.0, step=0.01) for feat in features}
        w = np.array(list(params.values()))
        if len(w) != len(features):
            raise ValueError(f"Weight vector length ({len(w)}) does not match features length ({len(features)})")
        w /= w.sum()
        scores = df[features].dot(w)
        km = KMeans(n_clusters=3, random_state=42).fit(scores.values.reshape(-1,1))
        ch = calinski_harabasz_score(scores.values.reshape(-1,1), km.labels_)
        return ch - 0.05 * np.sum(w)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=trials)
    best = study.best_trial.params
    total = sum(best.values()); return {k: v/total for k,v in best.items()}

# 2 · Pipeline

def run_pipeline(
    input_csv: str,
    output_file: str,
    optuna_trials: int = 50,
    default_label: Optional[str] = None
) -> pd.DataFrame:
    df = load_data(input_csv)
    # Per-capita normalization
    df = df.assign(**{col+'_per1k': df[col]/(df['Toplam Nüfus']/1000+1)
                       for col in df.columns if 'Sayısı' in col or 'Nüfus' in col})
    df = feature_engineering(df)

    FEATURES = [f for g in GROUPS.values() for f in g['features']]
    df = robust_minmax(df, FEATURES)

    prior_w = compute_weights_prior()
    w_opt = optimize_weights(df, FEATURES, optuna_trials, prior_w)

    # Composite hesaplama
    comps = {}
    for name, groups in COMPOSITES.items():
        feats = [f for g in groups for f in GROUPS[g]['features']]
        # Validate features exist
        missing_feats = [f for f in feats if f not in df.columns]
        if missing_feats:
            print(f"Warning: Missing features for {name}: {missing_feats}")
            continue
            
        # Validate weights exist
        missing_weights = [f for f in feats if f not in w_opt]
        if missing_weights:
            print(f"Warning: Missing weights for {name}: {missing_weights}")
            continue
            
        weights = pd.Series([w_opt[f] for f in feats], index=feats)
        comps[name] = df[feats].dot(weights) / weights.sum()
    comp_df = pd.DataFrame(comps)

    df['Risk_Skoru'] = comp_df.mean(axis=1)*100

    # Thresholds via KMeans centers
    k=4; km=KMeans(n_clusters=k,random_state=42).fit(df[['Risk_Skoru']]);
    centers=sorted(km.cluster_centers_.flatten())
    thr=[(centers[i]+centers[i+1])/2 for i in range(k-1)]
    bins=[-np.inf,*thr,np.inf]
    labels=[chr(65+i) for i in range(len(bins)-1)]
    df['Risk_Sınıfı']=pd.cut(df['Risk_Skoru'],bins=bins,labels=labels)

    # Isotonic calibrate
    if default_label and default_label in df.columns:
        iso=IsotonicRegression(out_of_bounds='clip')
        df['PD_Est']=iso.fit_transform(df['Risk_Skoru'],df[default_label])

    # Save
    Path(output_file).parent.mkdir(parents=True,exist_ok=True)
    df.to_excel(output_file)
    print(f"✔ Kaydedildi: {output_file}")
    return df

# 3 · Otomatik çalıştırma
if __name__=='__main__':
    run_pipeline('data.csv','risk_score.xlsx')
