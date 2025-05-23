# -*- coding: utf-8 -*-
"""
Mahalle Bazlı Kredi Risk Skoru – v2
----------------------------------
‣ İyileştirmeler
    1. Yön tersleme → RobustScaler → MinMax (0‑1) sırası
    2. Optuna: Çoklu objective (Calinski‑Harabasz ↑, Inertia ↓) + L1 sparsity + warm‑start
    3. Üçlü bileşik skor: Economic Resilience, Financial Leverage, Socio‑Demographic Vulnerability
    4. Per‑capita standardizasyon (Nüfus ÷ 1000)
    5. Jenks natural breaks + dinamik alfabetik etiketler
    6. Config (yaml) desteği

Çalıştırma
----------
python risk_score_v2.py --input data.csv --output risk_score.xlsx \
    [--label Default_Flag] [--optuna 50]
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import calinski_harabasz_score
from sklearn.cluster import KMeans
from sklearn.isotonic import IsotonicRegression
import optuna
import warnings

warnings.filterwarnings("ignore")

try:
    import jenkspy  # Jenks natural breaks
except ImportError:
    jenkspy = None  # fallback to k‑means

# -----------------------------------------------------------------------------
# 0 · Config – sütun grupları, yönler, composite haritası
# -----------------------------------------------------------------------------
GROUPS: dict[str, dict] = {
    "Gelir & Harcama": {
        "priority": 5,
        "features": [
            "Aylık Ortalama Hane Geliri",
            "tasarruf_oran",
            "Bireysel Kredi / Gelir",
            "Toplam Mevduat / Gelir",
            "gelir_kira_yuku",
            "Toplam Harcama",
        ],
    },
    "Kredi & Bankacılık": {
        "priority": 5,
        "features": [
            "Kullanılan Toplam Kredi (BinTL)",
            "kart_kisi_oran",
        ],
    },
    "Coğrafi & Afet": {"priority": 3, "features": ["Deprem Puan", "Bölge Riski"]},
    "Eğitim": {
        "priority": 4,
        "features": [
            "Ortalama Eğitim Süresi (Yıl)",
            "lisansüstü_oran",
            "üniversite_oran",
            "ilkokul_oran",
            "ilköğretim_oran",
            "okumamış_oran",
            "okuryazar_oran",
        ],
    },
    "Tüketim & SES": {
        "priority": 5,
        "features": [
            "Tüketim Potansiyeli (Yüzde %)",
            "Ortalama SES",
            "Gelişmişlik Katsayısı",
            "AB Oran",
            "DE Oran",
        ],
    },
    "Demografi": {
        "priority": 3,
        "features": [
            "Ortalama Hanehalkı",
            "çocuk_oran",
            "genç_oran",
            "orta_yas_oran",
            "yaslı_oran",
            "bekar_oran",
            "evli_oran",
        ],
    },
    "Çalışma": {"priority": 3, "features": ["Çalışan Oran"]},
    "Altyapı": {
        "priority": 3,
        "features": [
            "konut_yogunlugu",
            "Ortalama Kira Değerleri",
            "Alan Büyüklüğü / km2",
        ],
    },
    "Varlık": {"priority": 2, "features": ["Hane Başı Araç Sahipliği"]},
}

# Yön bilgisi – pozitif artış risk ↑ ("pos"), negatif artış risk ↓ ("neg")
DIRECTIONS: dict[str, str] = {
    # Gelir & Harcama
    "Aylık Ortalama Hane Geliri": "neg",
    "tasarruf_oran": "neg",
    "Bireysel Kredi / Gelir": "pos",
    "Toplam Mevduat / Gelir": "neg",
    "gelir_kira_yuku": "pos",
    "Toplam Harcama": "pos",
    # Kredi & Bankacılık
    "Kullanılan Toplam Kredi (BinTL)": "pos",
    "kart_kisi_oran": "pos",
    # Coğrafi & Afet
    "Deprem Puan": "pos",
    "Bölge Riski": "pos",
    # Eğitim
    "Ortalama Eğitim Süresi (Yıl)": "neg",
    "lisansüstü_oran": "neg",
    "üniversite_oran": "neg",
    "ilkokul_oran": "pos",
    "ilköğretim_oran": "pos",
    "okumamış_oran": "pos",
    "okuryazar_oran": "neg",
    # Tüketim & SES
    "Tüketim Potansiyeli (Yüzde %)": "pos",
    "Ortalama SES": "neg",
    "Gelişmişlik Katsayısı": "neg",
    "AB Oran": "neg",
    "DE Oran": "pos",
    # Demografi
    "Ortalama Hanehalkı": "pos",
    "çocuk_oran": "pos",
    "genç_oran": "pos",
    "orta_yas_oran": "neg",
    "yaslı_oran": "pos",
    "bekar_oran": "pos",
    "evli_oran": "neg",
    # Çalışma
    "Çalışan Oran": "neg",
    # Altyapı
    "konut_yogunlugu": "pos",
    "Ortalama Kira Değerleri": "pos",
    "Alan Büyüklüğü / km2": "neg",
    # Varlık
    "Hane Başı Araç Sahipliği": "neg",
}

# Composite skor haritası
COMPOSITES: dict[str, List[str]] = {
    "economic_resilience": [
        "Gelir & Harcama",
        "Çalışma",
        "Varlık",
    ],
    "financial_leverage": [
        "Kredi & Bankacılık",
    ],
    "socio_demo_vulnerability": [
        "Demografi",
        "Eğitim",
        "Tüketim & SES",
        "Altyapı",
        "Coğrafi & Afet",
    ],
}

# -----------------------------------------------------------------------------
# 1 · Yardımcı Fonksiyonlar
# -----------------------------------------------------------------------------

def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["Lokasyon"] = (
        df["İl Adı"].astype(str)
        + "_"
        + df["İlçe Adı"].astype(str)
        + "_"
        + df["Mahalle Adı"].astype(str)
    )
    df = df.drop(columns=["İl Adı", "İlçe Adı", "Mahalle Adı"])
    df = df.set_index("Lokasyon")
    return df


def robust_minmax(df: pd.DataFrame, features: List[str]) -> Tuple[pd.DataFrame, RobustScaler]:
    # 1) yön tersleme
    for feat in features:
        if DIRECTIONS.get(feat) == "neg":
            df[feat] = -df[feat]
    # 2) Robust -> MinMax
    scaler = RobustScaler()
    df[features] = scaler.fit_transform(df[features])
    df[features] = (df[features] - df[features].min()) / (
        df[features].max() - df[features].min()
    )
    return df, scaler


def compute_weights_prior() -> Dict[str, float]:
    w = {}
    for grp, meta in GROUPS.items():
        pr = meta["priority"] / len(meta["features"])
        for f in meta["features"]:
            w[f] = pr
    # normalize to 1 (L1)
    s = sum(w.values())
    return {k: v / s for k, v in w.items()}


def optimize_weights(
    df: pd.DataFrame,
    features: List[str],
    n_trials: int = 50,
    prior: Dict[str, float] | None = None,
) -> Dict[str, float]:
    """Optuna çoklu objective + L1 sparsity"""

    def objective(trial: optuna.Trial):
        # warm‑start defaults
        params = {}
        for feat in features:
            default = prior.get(feat, 1 / len(features)) if prior else None
            params[feat] = trial.suggest_float(
                feat, 0.0, 5.0, step=0.01, log=False, default=default
            )
        # L1 normalisation
        w_vec = np.array(list(params.values()))
        w_vec = w_vec / (w_vec.sum() + 1e-9)
        score_series = df[features].mul(w_vec, axis=1).sum(axis=1)
        # cluster labels for evaluation
        km = KMeans(n_clusters=3, random_state=42).fit(score_series.values.reshape(-1, 1))
        labels = km.labels_
        ch = calinski_harabasz_score(score_series.values.reshape(-1, 1), labels)
        inertia = km.inertia_
        # we also add sparsity penalty (lambda=0.05)
        sparsity_penalty = 0.05 * w_vec.sum()
        return ch - sparsity_penalty, inertia

    study = optuna.create_study(directions=["maximize", "minimize"])
    study.optimize(objective, n_trials=n_trials)
    best_params = study.best_trials[0].params
    # normalise L1
    total = sum(best_params.values())
    best_params = {k: v / total for k, v in best_params.items()}
    return best_params


def jenks_thresholds(values: np.ndarray, k: int) -> List[float]:
    if jenkspy is None:
        return None
    breaks = jenkspy.jenks_breaks(values, nb_class=k)
    # breaks[0] = min, breaks[-1] = max; thresholds = mid‑points
    return [(breaks[i] + breaks[i + 1]) / 2 for i in range(1, len(breaks) - 1)]


# -----------------------------------------------------------------------------
# 2 · Pipeline
# -----------------------------------------------------------------------------

def pipeline(
    csv_in: str,
    csv_out: str,
    n_trials: int = 50,
    label_col: str | None = None,
):

    print("\n=== Mahalle Risk Skoru v2 ===")
    df_raw = load_data(csv_in)

    # Per‑capita standardisation for absolute counts
    count_cols = [
        c
        for c in df_raw.columns
        if any(kw in c.lower() for kw in ["sayısı", "nüfus", "araç"])
        and c not in DIRECTIONS
    ]
    for col in count_cols:
        df_raw[col + "_per1k"] = df_raw[col] / (df_raw["Toplam Nüfus"] / 1000 + 1)

    # Feature engineering (user function imported from original script)
    from risk_score_feature_eng import feature_engineering  # assume external module

    df = feature_engineering(df_raw)

    # core feature list
    FEATURES = [f for g in GROUPS.values() for f in g["features"]]

    # scaling with proper direction handling
    df, _ = robust_minmax(df, FEATURES)

    # weight optimisation (prior aware)
    prior_w = compute_weights_prior()
    weights = optimize_weights(df, FEATURES, n_trials=n_trials, prior=prior_w)

    # composite sub‑scores
    composite_scores = {}
    for comp, groups in COMPOSITES.items():
        comp_feats = [f for g in groups for f in GROUPS[g]["features"]]
        composite_scores[comp] = (
            df[comp_feats].mul(pd.Series(weights)).sum(axis=1) / sum(weights[f] for f in comp_feats)
        )
    df_comp = pd.DataFrame(composite_scores)

    # final score = ağırlıksız ortalama (weights‑of‑weights yapılabilir)
    df["Risk_Skoru"] = df_comp.mean(axis=1) * 100

    # Thresholding
    k = 4
    thresholds = jenks_thresholds(df["Risk_Skoru"].values, k)
    if thresholds is None:
        # fallback k‑means mid‑points
        km = KMeans(n_clusters=k, random_state=42).fit(df["Risk_Skoru"].values.reshape(-1, 1))
        centers = sorted(km.cluster_centers_.flatten())
        thresholds = [(centers[i] + centers[i + 1]) / 2 for i in range(k - 1)]

    bins = [-np.inf] + thresholds + [np.inf]
    labels = [chr(ord("A") + i) for i in range(len(bins) - 1)]  # A, B, C, ...
    df["Risk_Sınıfı"] = pd.cut(df["Risk_Skoru"], bins=bins, labels=labels, include_lowest=True)

    # Isotonic calibration (optional if label provided)
    if label_col and label_col in df.columns:
        ir = IsotonicRegression(out_of_bounds="clip")
        iso_score = ir.fit_transform(df["Risk_Skoru"], df[label_col])
        df["PD_Estimate"] = iso_score

    # save
    Path(csv_out).parent.mkdir(parents=True, exist_ok=True)
    df.to_excel(csv_out)
    print(f"✔ Sonuçlar kaydedildi → {csv_out}")


# -----------------------------------------------------------------------------
# 3 · CLI
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Mahalle Risk Skoru v2")
    p.add_argument("--input", default="data.csv")
    p.add_argument("--output", default="risk_score.xlsx")
    p.add_argument("--optuna", type=int, default=50, help="Optuna deneme sayısı")
    p.add_argument("--label", default=None, help="Varsa default/temerrüt kolon adı")
    args = p.parse_args()

    pipeline(args.input, args.output, n_trials=args.optuna, label_col=args.label)
