#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
risk_segmentation_custom_loader.py

1. Excel'den veri yükler (input_path),
   - İl, İlçe, Mahalle sütunlarını 'Lokasyon' indexine dönüştürür,
   - Nüfusu 500'ün altındaki satırları atar.
2. GROUPS içindeki feature'ları filtreler ve eksik verileri çıkarır.
3. MinMax ölçekleme + neg/pos yön düzeltmesi.
4. Entropi tabanlı ağırlık hesaplar.
5. Log-multiplikatif risk skoru üretir ve [0-100] aralığına normlar.
6. Quantile segment ataması yapar.
7. Sonucu CSV'ye yazar, ağırlıkları terminale basar.

Kullanım:
    python risk_segmentation_custom_loader.py --input veri.xlsx --output results.csv
"""

import argparse
import sys
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import entropy as shannon_entropy

EPS = 1e-6

# 1. Veri Yükleme
# --------------------------------------------------------------------
def load_data(input_path: str) -> pd.DataFrame:
    # Excel okuma
    df = pd.read_excel(input_path)
    # Lokasyon oluşturma
    df["Lokasyon"] = (
        df["İl Adı"].astype(str) + "-" +
        df["İlçe Adı"].astype(str) + "_" +
        df["Mahalle Adı"].astype(str)
    )
    df.drop(columns=["İl Adı", "İlçe Adı", "Mahalle Adı"], inplace=True)
    df.set_index("Lokasyon", inplace=True)
    # Minimum nüfus filtresi
    if "Toplam Nüfus" in df.columns:
        df = df[df["Toplam Nüfus"] >= 500]
    return df

# 2. Gruplar & Yön Bilgileri
# --------------------------------------------------------------------
GROUPS = {
    "Gelir & Harcama": {"features": [
        "Aylık Ortalama Hane Geliri", "tasarruf_oran", "Bireysel Kredi / Gelir",
        "Toplam Mevduat / Gelir", "gelir_kira_yuku", "Toplam Harcama",
        "Max_Min_Income_Ratio", "Income_Gini_Proxy"
    ]},
    "Kredi & Bankacılık": {"features": ["Kullanılan Toplam Kredi (BinTL)", "kart_kisi_oran"]},
    "Coğrafi & Afet": {"features": ["Deprem Puan", "Bölge Riski"]},
    "Eğitim": {"features": [
        "Ortalama Eğitim Süresi (Yıl)", "lisansüstü_oran", "üniversite_oran",
        "ilkokul_oran", "ilköğretim_oran", "okunmamış_oran", "okuryazar_oran"
    ]},
    "Tüketim & SES": {"features": [
        "Tüketim Potansiyeli (Yüzde %)", "Ortalama SES", "Gelişmişlik Katsayısı",
        "AB Oran", "DE Oran"
    ]},
    "Demografi": {"features": [
        "Ortalama Hanehalkı", "çocuk_oran", "genç_oran", "orta_yas_oran",
        "yaslı_oran", "bekar_oran", "evli_oran", "Working_Age_Share", "Dependency_Ratio"
    ]},
    "Çalışma": {"features": ["Çalışan Oranı"]},
    "Altyapı": {"features": [
        "Konut Yoğunluğu (Konut/Km2)", "Ortalama Kira Değerleri",
        "Alan Büyüklüğü / km2", "Population_Density"
    ]},
    "Varlık": {"features": ["Hane Başı Araç Sahipliği"]}
}

# neg/pos yön
DIRECTIONS = {**{f: "neg" for f in [
    "Aylık Ortalama Hane Geliri","tasarruf_oran","Toplam Mevduat / Gelir",
    "Ortalama Eğitim Süresi (Yıl)","lisansüstü_oran","üniversite_oran",
    "okuryazar_oran","Ortalama SES","Gelişmişlik Katsayısı",
    "AB Oran","evli_oran","Working_Age_Share","Çalışan Oranı",
    "Alan Büyüklüğü / km2","Hane Başı Araç Sahipliği"
]]},
               **{f: "pos" for f in [
    "Bireysel Kredi / Gelir","gelir_kira_yuku","Toplam Harcama",
    "Max_Min_Income_Ratio","Income_Gini_Proxy","Kullanılan Toplam Kredi (BinTL)",
    "kart_kisi_oran","Deprem Puan","Bölge Riski","ilkokul_oran",
    "ilköğretim_oran","okunmamış_oran","Tüketim Potansiyeli (Yüzde %)",
    "DE Oran","Ortalama Hanehalkı","çocuk_oran","genç_oran",
    "orta_yas_oran","yaslı_oran","bekar_oran",
    "Konut Yoğunluğu (Konut/Km2)","Ortalama Kira Değerleri","Population_Density"
]]}

# 3. Pipeline
# --------------------------------------------------------------------
def pipeline(input_path: str, output_path: str):
    # Veri yükle
    df = load_data(input_path)

    # GROUPS içindeki feature'ları filtrele
    all_feats = [f for g in GROUPS.values() for f in g['features']]
    feats = [f for f in all_feats if f in df.columns]
    if not feats:
        sys.exit("❌ GROUPS içindeki hiçbir feature veri dosyasında bulunamadı.")

    # Eksikleri at
    df = df.dropna(subset=feats)

    # Ölçekleme
    df_s = df[feats].copy()
    df_s[feats] = MinMaxScaler().fit_transform(df_s[feats])
    # Yön düzeltme
    for f in feats:
        if DIRECTIONS.get(f) == "neg":
            df_s[f] = 1.0 - df_s[f]

    # Entropy ağırlıkları
    P = df_s / (df_s.sum(axis=0) + EPS)
    H = shannon_entropy(P, base=np.e, axis=0)
    d = 1 - H
    weights = {feats[i]: float(d[i]/d.sum()) for i in range(len(feats))}

    # Log-multiplikatif risk skoru
    ln_part = sum(weights[f] * np.log(df_s[f] + EPS) for f in feats)
    raw = np.exp(ln_part)
    df['RiskScore'] = 100 * (raw - raw.min()) / (raw.max() - raw.min())

    # Segmentasyon
    if df['RiskScore'].nunique() >= 3:
        df['Segment'] = pd.qcut(df['RiskScore'], q=3, labels=['Düşük','Orta','Yüksek'])
    else:
        df['Segment'] = 'Orta'

    # Kaydet + log
    df.to_csv(output_path)
    print(f"✓ Kullanılan feature sayısı: {len(feats)}")
    print("✓ Ağırlıklar:")
    for k,v in weights.items():
        print(f"   {k:30} → {v:.3f}")
    print(f"✓ RiskScore & Segment eklendi ve kaydedildi → {output_path}")

# 4. Argparse
# --------------------------------------------------------------------
if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--input',  required=True, help='Excel dosyası (İl, İlçe, Mahalle sütunlu)')
    p.add_argument('--output', required=True, help='Çıktı CSV dosyası')
    args = p.parse_args()
    pipeline(args.input, args.output)
