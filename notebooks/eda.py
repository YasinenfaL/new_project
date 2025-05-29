#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
risk_segmentation_custom_loader.py

Excel'den veri yükler, Lokasyon oluşturur, GROUPS içindeki feature'lara göre
ölçekleme ve yön düzeltme yapar. Manuel olarak verilen grup ağırlıklarıyla
ağırlıklı ortalama (dot product) yöntemiyle risk skoru oluşturur.

Kullanım:
    python risk_segmentation_custom_loader.py --input veri.xlsx --output results.csv
"""

import argparse
import sys
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

EPS = 1e-6

# 1. Veri Yükleme
# --------------------------------------------------------------------
def load_data(input_path: str) -> pd.DataFrame:
    df = pd.read_excel(input_path)
    df["Lokasyon"] = (
        df["İl Adı"].astype(str) + "-" +
        df["İlçe Adı"].astype(str) + "_" +
        df["Mahalle Adı"].astype(str)
    )
    df.drop(columns=["İl Adı", "İlçe Adı", "Mahalle Adı"], inplace=True)
    df.set_index("Lokasyon", inplace=True)
    if "Toplam Nüfus" in df.columns:
        df = df[df["Toplam Nüfus"] >= 500]
    return df

# 2. Gruplar & Ağırlıklar
# --------------------------------------------------------------------
GROUPS = {
    "Gelir & Harcama": {"weight": 0.20, "features": [
        "Aylık Ortalama Hane Geliri", "tasarruf_oran", "Bireysel Kredi / Gelir",
        "Toplam Mevduat / Gelir", "gelir_kira_yuku", "Toplam Harcama",
        "Max_Min_Income_Ratio", "Income_Gini_Proxy"]},
    "Kredi & Bankacılık": {"weight": 0.10, "features": ["Kullanılan Toplam Kredi (BinTL)", "kart_kisi_oran"]},
    "Coğrafi & Afet": {"weight": 0.10, "features": ["Deprem Puan", "Bölge Riski"]},
    "Eğitim": {"weight": 0.15, "features": [
        "Ortalama Eğitim Süresi (Yıl)", "lisansüstü_oran", "üniversite_oran",
        "ilkokul_oran", "ilköğretim_oran", "okunmamış_oran", "okuryazar_oran"]},
    "Tüketim & SES": {"weight": 0.10, "features": [
        "Tüketim Potansiyeli (Yüzde %)", "Ortalama SES", "Gelişmişlik Katsayısı",
        "AB Oran", "DE Oran"]},
    "Demografi": {"weight": 0.15, "features": [
        "Ortalama Hanehalkı", "çocuk_oran", "genç_oran", "orta_yas_oran",
        "yaslı_oran", "bekar_oran", "evli_oran", "Working_Age_Share", "Dependency_Ratio"]},
    "Çalışma": {"weight": 0.05, "features": ["Çalışan Oranı"]},
    "Altyapı": {"weight": 0.10, "features": [
        "Konut Yoğunluğu (Konut/Km2)", "Ortalama Kira Değerleri",
        "Alan Büyüklüğü / km2", "Population_Density"]},
    "Varlık": {"weight": 0.05, "features": ["Hane Başı Araç Sahipliği"]}
}

DIRECTIONS = {
    **{f: "neg" for f in [
        "Aylık Ortalama Hane Geliri","tasarruf_oran","Toplam Mevduat / Gelir",
        "Ortalama Eğitim Süresi (Yıl)","lisansüstü_oran","üniversite_oran",
        "okuryazar_oran","Ortalama SES","Gelişmişlik Katsayısı",
        "AB Oran","evli_oran","Working_Age_Share","Çalışan Oranı",
        "Alan Büyüklüğü / km2","Hane Başı Araç Sahipliği"]},
    **{f: "pos" for f in [
        "Bireysel Kredi / Gelir","gelir_kira_yuku","Toplam Harcama",
        "Max_Min_Income_Ratio","Income_Gini_Proxy","Kullanılan Toplam Kredi (BinTL)",
        "kart_kisi_oran","Deprem Puan","Bölge Riski","ilkokul_oran",
        "ilköğretim_oran","okunmamış_oran","Tüketim Potansiyeli (Yüzde %)",
        "DE Oran","Ortalama Hanehalkı","çocuk_oran","genç_oran",
        "orta_yas_oran","yaslı_oran","bekar_oran",
        "Konut Yoğunluğu (Konut/Km2)","Ortalama Kira Değerleri","Population_Density"]}

# 3. Pipeline
# --------------------------------------------------------------------
def pipeline(input_path: str, output_path: str):
    df = load_data(input_path)

    feats = [f for g in GROUPS.values() for f in g['features'] if f in df.columns]
    df = df.dropna(subset=feats)

    df_s = df[feats].copy()
    df_s[feats] = MinMaxScaler().fit_transform(df_s[feats])
    for f in feats:
        if DIRECTIONS.get(f) == "neg":
            df_s[f] = 1.0 - df_s[f]

    # Grup bazlı ağırlıklı skor
    score = pd.Series(0, index=df.index, dtype=float)
    for group, meta in GROUPS.items():
        group_feats = [f for f in meta['features'] if f in df_s.columns]
        if group_feats:
            group_avg = df_s[group_feats].mean(axis=1)
            score += meta['weight'] * group_avg

    df['RiskScore'] = 100 * (score - score.min()) / (score.max() - score.min())

    if df['RiskScore'].nunique() >= 3:
        df['Segment'] = pd.qcut(df['RiskScore'], q=3, labels=['Düşük','Orta','Yüksek'])
    else:
        df['Segment'] = 'Orta'

    df.to_csv(output_path)
    print("✓ Grup ağırlıkları ile RiskScore hesaplandı.")
    print("✓ Segment ataması yapıldı ve sonuç kaydedildi.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input',  required=True, help='Excel dosyası')
    parser.add_argument('--output', required=True, help='Çıktı CSV dosyası')
    args = parser.parse_args()
    pipeline(args.input, args.output)


Kod başarıyla güncellendi. Artık risk skoru, her bir GROUP için manuel belirlenmiş ağırlıklarla ve grup ortalamaları üzerinden hesaplanıyor. Bu daha kontrollü ve açıklanabilir bir yöntemdir. Artık çalıştırarak gerçek verinizle sonucu alabilirsiniz. Hazırsanız test edebiliriz.

