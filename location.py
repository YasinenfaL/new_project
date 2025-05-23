from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler

import warnings
warnings.filterwarnings('ignore')
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=RuntimeWarning)


CSV_IN: str = "data.csv"              # Girdi CSV yolunu ayarla
CSV_OUT: str = "risk_score.csv"   # Çıktı CSV yolunu ayarla


# ---------------------------------------------------------------------------
# 1 · Feature Grupları & Öncelikler
# ---------------------------------------------------------------------------
GROUPS: dict[str, dict] = {
    "Gelir & Harcama": {
        "priority": 5,
        "features": [
            "Aylık Ortalama Hane Geliri",
            "tasarruf_oran",               # Aylık Tasarruf / Gelir
            "Bireysel Kredi / Gelir",      # önceden hazırlanmış oran kolonu olmalı
            "Toplam Mevduat / Gelir",      # önceden hazırlanmış oran kolonu olmalı
            "gelir_kira_yuku",
            "Toplam Harcama",
        ],
    },
    "Kredi & Bankacılık": {
        "priority": 5,
        "features": ["Kullanılan Toplam Kredi (BinTL)", "kart_kisi_oran"],
    },
    "Coğrafi & Afet": {
        "priority": 3,
        "features": ["Deprem Puan", "Bölge Riski"],
    },
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


# ─────────────────────────────────────────────────────────────────────────────
# 1.  I/O
# ─────────────────────────────────────────────────────────────────────────────
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["Lokasyon"] = (
        df["İl Adı"].astype(str) + "_" +
        df["İlçe Adı"].astype(str) + "_" +
        df["Mahalle Adı"].astype(str)
    )
    df.drop(columns=["İl Adı", "İlçe Adı", "Mahalle Adı"], inplace=True)
    df.set_index("Lokasyon", inplace=True)
    return df


def save_data(df: pd.DataFrame, path: str) -> None:
    df.to_csv(path, index=True)
    print(f"✔  Sonuçlar «{path}» dosyasına kaydedildi.")


# ─────────────────────────────────────────────────────────────────────────────
# 2.  FEATURE ENGINEERING  (türetilmiş sütunlar)
# ─────────────────────────────────────────────────────────────────────────────
def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Demografik oranlar
    df["erkek_oran"] = df["Erkek Nüfusu"] / df["Toplam Nüfus"]
    df["kadın_oran"] = df["Kadın Nüfusu"] / df["Toplam Nüfus"]
    df["çocuk_oran"] = df["0-15 Kişi Sayısı"] / df["Toplam Nüfus"]
    df["genç_oran"] = df["15-25 Kişi Sayısı"] / df["Toplam Nüfus"]
    df["orta_yas_oran"] = df[["25-40 Kişi Sayısı", "40-55 Kişi Sayısı"]].sum(axis=1) / df["Toplam Nüfus"]
    df["yaslı_oran"] = df["55+ Kişi Sayısı"] / df["Toplam Nüfus"]

    # Eğitim oranları
    df["lisansüstü_oran"] = (df["Yüksek Lisans Kişi Sayısı"] + df["Doktora Kişi Sayısı"]) / df["Toplam Nüfus"]
    df["üniversite_oran"] = df["Üniversite ve Üstü Mezun"] / df["Toplam Nüfus"]
    df["ilkokul_oran"] = df["İlkokul Kişi Sayısı"] / df["Toplam Nüfus"]
    df["ilköğretim_oran"] = df["İlköğretim Kişi Sayısı"] / df["Toplam Nüfus"]
    df["okumamış_oran"] = df["Okumamış Kişi Sayısı"] / df["Toplam Nüfus"]
    df["okuryazar_oran"] = df["Okuryazar Kişi Sayısı"] / df["Toplam Nüfus"]

    # Aile yapısı ve medeni durum
    df["bekar_oran"] = df["Bekar Kişi Sayısı"] / df["Toplam Nüfus"]
    df["evli_oran"] = df["Evli Kişi Sayısı"] / df["Toplam Nüfus"]

    # Gelir – harcama ilişkisi
    df["tasarruf_oran"] = df["Aylık Hane Tasarrufu"] / (df["Aylık Ortalama Hane Geliri"] + 1)
    df["harcama_oran"] = df["Aylık Ortalama Hane Harcaması"] / (df["Aylık Ortalama Hane Geliri"] + 1)
    df["gelir_kira_yuku"] = df["Ortalama Kira Değerleri"] / (df["Aylık Ortalama Hane Geliri"] + 1)

    # Ekonomik yük – hane & kredi
    df["hane_kredi_oran"] = df["Kullanılan Bireysel Kredi (BinTL"] / (df["Toplam Mevduat (Bin TL"] + 1)
    df["kurumsal_kredi_oran"] = df["Kullanılan Kurumsal Kredi (BinTL"] / (df["Toplam Mevduat (Bin TL"] + 1)
    df["tasarruf_gsyh_oran"] = df["Aylık Hane Tasarrufu"] / (df["2019 yılı GSYH (TL"] + 1)

    # Yerleşim yapısı ve yoğunluk
    df["konut_yogunlugu"] = df["Konut Sayısı"] / (df["Alan Büyüklüğü / km"] + 1)
    df["işyeri_yogunlugu"] = df["İş Yeri Sayısı"] / (df["Alan Büyüklüğü / km"] + 1)
    df["konut_ticaret_oran"] = df["Bölge Konutlaşma Oran ( Yüzde"] / (df["Bölge Ticaretleşme Oran ( Yüzde"] + 1)

    # Ulaşım ve mobilite
    df["araç_kisi_oran"] = df["Toplam Araç Sayısı (Adet"] / (df["Toplam Nüfus"] + 1)
    df["elektrikli_arac_oran"] = df["Elektrikli Araç Sayısı (Adet"] / (df["Toplam Araç Sayısı (Adet"] + 1)

    # Kredi kartı yaygınlığı
    df["kart_kisi_oran"] = (df["Banka Kartı Sayısı"] + df["Kredi Kartı Sayısı"] + df["Ön Ödemeli Kart Sayısı"]) / (df["Toplam Nüfus"] + 1)

    # Eğitimde kalış süresi
    df["eğitim_suresi_yas_oran"] = df["Ortalama Eğitim Süresi (Yıl"] / ((df["Toplam Nüfus"] / 4) + 1)  # kabaca eğitim yaş grubu

    df.drop(columns=["Erkek Nüfusu",
                 "Kadın Nüfusu",
                 "0-15 Kişi Sayısı",
                 "15-25 Kişi Sayısı",
                 "25-40 Kişi Sayısı",
                 "40-55 Kişi Sayısı",
                 "55+ Kişi Sayısı",
                 "Yüksek Lisans Kişi Sayısı",
                 "Doktora Kişi Sayısı",
                 "Üniversite ve Üstü Mezun",
                 "İlkokul Kişi Sayısı",
                 "İlköğretim Kişi Sayısı",
                 "Okumamış Kişi Sayısı",
                 "Okuryazar Kişi Sayısı",
                 "Bekar Kişi Sayısı",
                 "Evli Kişi Sayısı",
                 "Fay Puan",
                 ],inplace=True,axis=1)
    df = df.replace([np.inf, -np.inf], 0)
    df = df.fillna(0)


    return df

# ---------------------------------------------------------------------------
# 3 · Ölçekleme & Ağırlıklar
# ---------------------------------------------------------------------------

def robust_scale(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    scaler = RobustScaler()
    df[cols] = scaler.fit_transform(df[cols])
    df[cols] = (df[cols]-df[cols].min())/(df[cols].max()-df[cols].min())
    return df


def compute_weights() -> dict[str,float]:
    w:dict[str,float]={}
    for grp,meta in GROUPS.items():
        weight = meta["priority"]/len(meta["features"])
        for feat in meta["features"]:
            w[feat] = weight
    return w


def weighted_risk(df: pd.DataFrame,w:dict[str,float]) -> pd.Series:
    total = sum(w.values())
    return 100*df[list(w)].mul(pd.Series(w)).sum(axis=1)/total

# ---------------------------------------------------------------------------
# 4 · Pipeline
# ---------------------------------------------------------------------------

def pipeline():
    df=load_data(CSV_IN)
    df=feature_engineering(df)

    core=[f for grp in GROUPS.values() for f in grp["features"]]
    miss=[c for c in core if c not in df.columns]
    if miss: raise ValueError(f"Eksik kolonlar: {miss}")

    df=robust_scale(df,core)
    w=compute_weights()
    df["Risk_Skoru"]=weighted_risk(df,w)
    df["Risk_Sınıfı"]=pd.cut(df["Risk_Skoru"],bins=[-np.inf,35,65,np.inf],labels=["Low-Risk","Mid-Risk","High-Risk"])

    Path(CSV_OUT).parent.mkdir(exist_ok=True,parents=True)
    df.to_csv(CSV_OUT)
    print(f"✔ Risk skorları kaydedildi → {CSV_OUT}")

if __name__=="__main__":
    pipeline()
