Aşağıda tüm parçaları (dosya yükleme, GROUPS, DIRECTIONS, ölçekleme, yön düzeltme, entropi tabanlı ağırlık, log-multiplikatif risk skoru ve quantile segmentasyon) bir arada görebileceğiniz, çalıştırmaya hazır bir Python betiği var. Kendi yolunuza göre --csv_in / --csv_out argümanlarını verip kullanabilirsiniz.

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
risk_segmentation_full.py

Kullanım:
    python risk_segmentation_full.py \
        --input lokasyon_data.csv \
        --output results.csv
"""

import argparse
import warnings
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import entropy as shannon_entropy

warnings.filterwarnings("ignore")
EPS = 1e-6

# --------------------------------------------------------------------
# 1 · Dosya Okuma (CSV ya da Excel otomatik seçer)
# --------------------------------------------------------------------
def load_data(path: str) -> pd.DataFrame:
    try:
        return pd.read_csv(path, index_col="Lokasyon")
    except Exception:
        return pd.read_excel(path, engine="openpyxl", index_col="Lokasyon")

# --------------------------------------------------------------------
# 2 · Feature Grupları & Yön Bilgileri
# --------------------------------------------------------------------
GROUPS = {
    "Gelir & Harcama": {"priority": 5, "features": [
        "Aylık Ortalama Hane Geliri", "tasarruf_oran", "Bireysel Kredi / Gelir",
        "Toplam Mevduat / Gelir", "gelir_kira_yuku", "Toplam Harcama",
        "Max_Min_Income_Ratio", "Income_Gini_Proxy"
    ]},
    "Kredi & Bankacılık": {"priority": 3, "features": [
        "Kullanılan Toplam Kredi (BinTL)", "kart_kisi_oran"
    ]},
    "Coğrafi & Afet": {"priority": 4, "features": [
        "Deprem Puan", "Bölge Riski"
    ]},
    "Eğitim": {"priority": 5, "features": [
        "Ortalama Eğitim Süresi (Yıl)", "lisansüstü_oran", "üniversite_oran",
        "ilkokul_oran", "ilköğretim_oran", "okumamış_oran", "okuryazar_oran"
    ]},
    "Tüketim & SES": {"priority": 5, "features": [
        "Tüketim Potansiyeli (Yüzde %)", "Ortalama SES", "Gelişmişlik Katsayısı",
        "AB Oran", "DE Oran"
    ]},
    "Demografi": {"priority": 5, "features": [
        "Ortalama Hanehalkı", "çocuk_oran", "genç_oran", "orta_yas_oran",
        "yaslı_oran", "bekar_oran", "evli_oran", "Working_Age_Share",
        "Dependency_Ratio"
    ]},
    "Çalışma": {"priority": 5, "features": [
        "Çalışan Oranı"
    ]},
    "Altyapı": {"priority": 5, "features": [
        "Konut Yoğunluğu (Konut/Km2)", "Ortalama Kira Değerleri",
        "Alan Büyüklüğü / km2", "Population_Density"
    ]},
    "Varlık": {"priority": 4, "features": [
        "Hane Başı Araç Sahipliği"
    ]},
}

DIRECTIONS = {
    # "neg" = yüksekçe düştükçe risk artar → ölçek sonrası 1-x
    # "pos" = yükseldikçe risk artar → ölçek sonrası direkt kullan
    "Aylık Ortalama Hane Geliri": "neg",
    "tasarruf_oran":               "neg",
    "Bireysel Kredi / Gelir":      "pos",
    "Toplam Mevduat / Gelir":      "neg",
    "gelir_kira_yuku":             "pos",
    "Toplam Harcama":              "pos",
    "Max_Min_Income_Ratio":        "pos",
    "Income_Gini_Proxy":           "pos",
    "Kullanılan Toplam Kredi (BinTL)": "pos",
    "kart_kisi_oran":              "pos",
    "Deprem Puan":                 "pos",
    "Bölge Riski":                 "pos",
    "Ortalama Eğitim Süresi (Yıl)": "neg",
    "lisansüstü_oran":             "neg",
    "üniversite_oran":             "neg",
    "ilkokul_oran":                "pos",
    "ilköğretim_oran":             "pos",
    "okumamış_oran":               "pos",
    "okuryazar_oran":              "neg",
    "Tüketim Potansiyeli (Yüzde %)": "pos",
    "Ortalama SES":                "neg",
    "Gelişmişlik Katsayısı":       "neg",
    "AB Oran":                     "neg",
    "DE Oran":                     "pos",
    "Ortalama Hanehalkı":          "pos",
    "çocuk_oran":                  "pos",
    "genç_oran":                   "pos",
    "orta_yas_oran":               "pos",
    "yaslı_oran":                  "pos",
    "bekar_oran":                  "pos",
    "evli_oran":                   "neg",
    "Working_Age_Share":           "neg",
    "Dependency_Ratio":            "pos",
    "Çalışan Oranı":               "neg",
    "Konut Yoğunluğu (Konut/Km2)": "pos",
    "Ortalama Kira Değerleri":     "pos",
    "Alan Büyüklüğü / km2":        "neg",
    "Population_Density":          "pos",
    "Hane Başı Araç Sahipliği":    "neg",
}

# --------------------------------------------------------------------
# 3 · Ölçekleme & Yön Düzeltme
# --------------------------------------------------------------------
def scale_and_direct(df: pd.DataFrame) -> (pd.DataFrame, list):
    feats = [f for grp in GROUPS.values() for f in grp["features"] if f in df.columns]
    df2 = df[feats].copy()
    df2[feats] = MinMaxScaler().fit_transform(df2[feats])
    for f in feats:
        if DIRECTIONS.get(f) == "neg":
            df2[f] = 1.0 - df2[f]
    return df2, feats

# --------------------------------------------------------------------
# 4 · Entropi Tabanlı Ağırlıklar
# --------------------------------------------------------------------
def compute_entropy_weights(df: pd.DataFrame, feats: list) -> dict:
    P = df[feats] / (df[feats].sum(axis=0) + EPS)
    H = shannon_entropy(P, base=np.e, axis=0)
    d = 1 - H
    total = d.sum()
    return {feats[i]: float(d[i] / total) for i in range(len(feats))}

# --------------------------------------------------------------------
# 5 · Log-Multiplikatif Risk Skoru
# --------------------------------------------------------------------
def compute_risk_score(df: pd.DataFrame, weights: dict) -> pd.Series:
    ln_part = sum(weights[f] * np.log(df[f] + EPS) for f in weights)
    raw = np.exp(ln_part)
    return 100 * (raw - raw.min()) / (raw.max() - raw.min())

# --------------------------------------------------------------------
# 6 · Quantile Segmentasyon
# --------------------------------------------------------------------
def assign_segment(df: pd.DataFrame, col: str, q: int = 3) -> pd.Series:
    return pd.qcut(df[col], q=q, labels=[f"Seg{i+1}" for i in range(q)])

# --------------------------------------------------------------------
# 7 · Pipeline
# --------------------------------------------------------------------
def pipeline(input_path: str, output_path: str):
    df = load_data(input_path)
    df_scaled, feats = scale_and_direct(df)

    weights = compute_entropy_weights(df_scaled, feats)
    df["RiskScore"] = compute_risk_score(df_scaled, weights)
    df["Segment"]   = assign_segment(df, "RiskScore", q=3)

    df.to_csv(output_path)
    print(f"✓ RiskScore ve Segment kolonları eklendi, {output_path} olarak kaydedildi.")
    print("— Ağırlıklar:")
    for k,v in weights.items():
        print(f"   {k:30} → {v:.3f}")

# --------------------------------------------------------------------
# 8 · Argparse & Çalıştırma
# --------------------------------------------------------------------
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--input",  required=True, help="Girdi CSV veya Excel dosyası (Lokasyon indexli)")
    p.add_argument("--output", required=True, help="Çıktı CSV dosyası")
    args = p.parse_args()

    pipeline(args.input, args.output)

Nasıl çalışır?

1. --input ile verdiğiniz dosyayı önce CSV, hata alırsa Excel (openpyxl) olarak okur.


2. GROUPS içindeki özellikleri MinMax ölçekler, DIRECTIONS ile “neg/pos” yön düzeltmesini yapar.


3. Entropi metodu ile değişken ağırlıklarını hesaplar.


4. Log-multiplikatif formülle 0–100 aralığında RiskScore üretir.


5. pd.qcut ile üç eşit gözlemli segment (Seg1/Seg2/Seg3) atar.


6. Sonucu CSV’ye yazar ve konsola ağırlıkları basar.



Herhangi bir ek modül (TOPSIS/VIKOR, farklı segment sayısı, harita görselleştirme vb.) isterseniz bu iskeleti temel alarak hızlıca ekleyebilirsiniz.

