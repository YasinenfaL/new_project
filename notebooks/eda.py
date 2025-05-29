#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
lokasyon_risk_v3.py — MinMax Scaler, Topsis Risk Skoru, 0–100 Ölçek
"""
import pandas as pd
import numpy as np
from scipy.stats.mstats import winsorize
from sklearn.preprocessing import MinMaxScaler
from numpy.linalg import norm

# ─────────────────────────────────────────────────────────────────────────────
# Yardımcı Fonksiyonlar
# ─────────────────────────────────────────────────────────────────────────────
def safe_div(numer, denom):
    """Sıfıra bölünmeyi engelle (0/0 → 0)."""
    denom = denom.replace(0, np.nan)
    return (numer / denom).fillna(0)


def entropy_weights(X: pd.DataFrame) -> pd.Series:
    """
    Entropi tabanlı bilgi-ağırlığı.
    X : 0–1 ölçekli, yön uygulanmış DataFrame.
    """
    P = X.abs().div(X.abs().sum(axis=0) + 1e-9, axis=1)
    n = len(X)
    k = 1.0 / np.log(n + 1e-9)
    ej = -k * (P * np.log(P + 1e-9)).sum(axis=0)
    dj = 1.0 - ej
    return dj / dj.sum()

# ─────────────────────────────────────────────────────────────────────────────
# Ana Akış
# ─────────────────────────────────────────────────────────────────────────────
def main():
    # 1) Veri Yükle ve Lokasyon Anahtarı
    df = pd.read_excel("veri.xlsx")
    df["Lokasyon"] = (
        df["İl"].str.strip() + "_" +
        df["İlçe"].str.strip() + "_" +
        df["Mahalle"].str.strip()
    )

    # 2) Feature-Engineering
    df_feat = df.copy()
    df_feat["depend_ratio"]        = safe_div(df["0-15 Kişi Sayısı"] + df["55+ Kişi Sayısı"],
                                                    df["15-25 Kişi Sayısı"] + df["25-40 Kişi Sayısı"] + df["40-55 Kişi Sayısı"])
    df_feat["female_share"]        = safe_div(df["Kadın Nüfusu"], df["Toplam Nüfus"])
    df_feat["syria_share"]         = safe_div(df["Geçici Koruma Kapsamındaki Suriyeli Sayısı"], df["Toplam Nüfus"])
    df_feat["foreign_share"]       = safe_div(df["İkamet Eden Yabancı"], df["Toplam Nüfus"])
    df_feat["income_per_house"]    = safe_div(df["Aylık Ortalama Hane Geliri"], df["Hanehalkı Sayısı"])
    df_feat["saving_rate"]         = safe_div(df["Aylık Hane Tasarrufu"], df["Aylık Ortalama Hane Harcaması"])
    df_feat["spending_propensity"] = safe_div(df["Aylık Ortalama Hane Harcaması"], df["Aylık Ortalama Hane Geliri"])
    df_feat["credit_per_capita"]   = safe_div(df["Bireysel Kredi (Bin TL)"], df["Toplam Nüfus"])
    df_feat["card_per_adult"]      = safe_div(df["Kredi Kartı Sayısı"], df["15-55 Kişi Sayısı"])
    df_feat["deposit_per_capita"]  = safe_div(df["Toplam Mevduat (BinTL)"], df["Toplam Nüfus"])
    df_feat["loan_to_deposit"]     = safe_div(df["Bireysel Kredi (Bin TL)"], df["Toplam Mevduat (BinTL)"])
    df_feat["employment_rate"]     = df["Çalışan Oran"]
    df_feat["firm_density"]        = safe_div(df["İş Yeri Sayısı Kent (Adet)"], df["Toplam Nüfus"])
    df_feat["atm_credit_ratio"]    = safe_div(df["Banka Kartı Sayısı"] + df["Kredi Kartı Sayısı"], df["Toplam Nüfus"])
    df_feat["uni_rate"]            = safe_div(df["Üniversitede Okuyan Öğrenci sayısı"], df["Toplam Nüfus"])
    df_feat["literacy_inv"]        = 1 - safe_div(df["Okur Yazarlık Sayısı"], df["Toplam Nüfus"])
    df_feat["household_per_house"] = safe_div(df["Hanehalkı Sayısı"], df["Konut Sayısı"])
    df_feat["vacancy_proxy"]       = safe_div(df["Konut Sayısı"] - df["Hanehalkı Sayısı"], df["Konut Sayısı"])

    FEATURES = [
        "depend_ratio","female_share","syria_share","foreign_share",
        "income_per_house","saving_rate","spending_propensity",
        "credit_per_capita","card_per_adult","deposit_per_capita","loan_to_deposit",
        "employment_rate","firm_density","atm_credit_ratio",
        "uni_rate","literacy_inv","household_per_house","vacancy_proxy",
    ]

    DIRECTION = {col: +1 for col in FEATURES}
    for col in ["income_per_house","saving_rate","deposit_per_capita","employment_rate","uni_rate"]:
        DIRECTION[col] = -1

    for col in FEATURES:
        df_feat[col] = winsorize(df_feat[col], limits=[0.01, 0.01])

    scaler = MinMaxScaler()
    X_scaled = pd.DataFrame(
        scaler.fit_transform(df_feat[FEATURES]),
        columns=FEATURES,
        index=df_feat.index
    )

    X_dir = X_scaled.mul(pd.Series(DIRECTION), axis=1)

    w_raw = entropy_weights(X_dir)

    # TOPSIS tabanlı risk skoru
    ideal     = X_dir.max()
    antiideal = X_dir.min()
    d_plus  = norm(X_dir - ideal, axis=1)
    d_minus = norm(X_dir - antiideal, axis=1)
    df_feat["risk_score"] = (d_plus / (d_plus + d_minus + 1e-9)) * 100

    lokasyon_risk = (
        df_feat.groupby("Lokasyon")["risk_score"]
               .mean()
               .reset_index(name="risk_score")
    )

    lokasyon_risk["risk_group"] = pd.qcut(
        lokasyon_risk["risk_score"], q=4,
        labels=["Düşük","Orta-Düşük","Orta-Yüksek","Yüksek"]
    )

    lokasyon_risk.to_csv("lokasyon_risk_scores_v3.csv", index=False)
    print("--- En Riskli 10 Lokasyon ---")
    print(lokasyon_risk.nlargest(10, "risk_score"))

if __name__ == "__main__":
    main()


Kodun “risk skoru” kısmı artık TOPSIS (vektörel mesafe tabanlı) yönteme göre hesaplanıyor. Skorlar 0–100 arasında normalize ediliyor ve 4 segmente ayrılıyor. Hazırsan bir sonraki adımda bu skorları harita, grafik veya stratejik raporlamaya taşıyabiliriz. Devam edelim mi?

