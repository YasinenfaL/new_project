from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import (
    KMeans, SpectralClustering, AgglomerativeClustering, DBSCAN, OPTICS
)
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import RobustScaler

# ─────────────────────────────────────────────────────────────────────────────
# Sabitler – ihtiyaca göre güncelle
# ─────────────────────────────────────────────────────────────────────────────
CSV_IN: str = "data.csv"              # Girdi CSV yolunu ayarla
CSV_OUT: str = "risk_segmented.csv"   # Çıktı CSV yolunu ayarla
VAR_RATIO: float = 0.95                # PCA toplam varyans eşiği
N_COMPONENTS: Optional[int] = None     # Sabit PCA bileşen sayısı (None = otomatik)

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
                 ])


    return df

# ─────────────────────────────────────────────────────────────────────────────
# 3 · Ölçekleme
# ─────────────────────────────────────────────────────────────────────────────

def scale_numeric(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    scaler = RobustScaler()
    df[cols] = scaler.fit_transform(df[cols])
    return df

# ─────────────────────────────────────────────────────────────────────────────
# 4 · PCA indirgeme
# ─────────────────────────────────────────────────────────────────────────────

def pca_reduce(
    df_num: pd.DataFrame,
    *, var_ratio: float,
    n_components: Optional[int]
) -> Tuple[pd.DataFrame, PCA]:
    pca = PCA(n_components=n_components if n_components else var_ratio, random_state=42)
    comps = pca.fit_transform(df_num)
    pc_cols = [f"PC{i+1}" for i in range(comps.shape[1])]
    return pd.DataFrame(comps, index=df_num.index, columns=pc_cols), pca

# ─────────────────────────────────────────────────────────────────────────────
# 5 · Kümeleme yardımcıları
# ─────────────────────────────────────────────────────────────────────────────

def choose_k(X: pd.DataFrame, rng=range(2, 9)) -> int:
    best_k, best_s = 2, -1
    for k in rng:
        labels = KMeans(k, n_init=10, random_state=42).fit_predict(X)
        sc = silhouette_score(X, labels)
        if sc > best_s:
            best_k, best_s = k, sc
    return best_k


def assign_risk(summary: pd.DataFrame) -> Dict[int, str]:
    rank = summary.rank(pct=True).mean(axis=1)
    idx = rank.sort_values().index.tolist()
    mapping: Dict[int, str] = {}
    if len(idx) == 1:
        mapping[idx[0]] = "Mid-Risk"
    elif len(idx) == 2:
        mapping[idx[0]], mapping[idx[1]] = "High-Risk", "Low-Risk"
    else:
        mapping[idx[0]], mapping[idx[-1]] = "High-Risk", "Low-Risk"
        for i in idx[1:-1]: mapping[i] = "Mid-Risk"
    return mapping


def cluster_and_profile(df: pd.DataFrame, features: List[str]) -> pd.DataFrame:
    X = df[features].astype(float)
    k_opt = choose_k(X)
    algos_k = {
        "kmeans": KMeans(k_opt, n_init=10, random_state=42),
        "spectral": SpectralClustering(k_opt, assign_labels="kmeans", random_state=42),
        "gmm": GaussianMixture(n_components=k_opt, random_state=42),
        "agg": AgglomerativeClustering(n_clusters=k_opt, linkage="ward"),
    }
    for name, m in algos_k.items():
        df[f"{name}_id"] = m.fit_predict(X)
        summ = df.groupby(f"{name}_id")[features].median()
        df[f"{name}_risk"] = df[f"{name}_id"].map(assign_risk(summ))

    density = {
        "dbscan": DBSCAN(eps=0.8, min_samples=10),
        "optics": OPTICS(min_samples=10, xi=0.05, min_cluster_size=0.02),
    }
    for name, m in density.items():
        labels = m.fit_predict(X)
        df[f"{name}_id"] = labels
        mask = labels != -1
        if mask.any():
            summ = df[mask].groupby(f"{name}_id")[features].median()
            mapper = assign_risk(summ)
            df[f"{name}_risk"] = df[f"{name}_id"].map(mapper)
            df.loc[~mask, f"{name}_risk"] = "Noise"
        else:
            df[f"{name}_risk"] = "Noise"
    return df

# ─────────────────────────────────────────────────────────────────────────────
# 6 · Görselleştirme
# ─────────────────────────────────────────────────────────────────────────────

def visualize_counts(df: pd.DataFrame, cluster_col: str, risk_col: str, title: str):
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    sns.countplot(x=df[cluster_col], ax=ax[0]); ax[0].set_title(f"{title} Clusters")
    sns.countplot(x=df[risk_col], order=["Low-Risk","Mid-Risk","High-Risk","Noise"], ax=ax[1]); ax[1].set_title(f"{title} Risk")
    plt.tight_layout(); plt.show()

# ─────────────────────────────────────────────────────────────────────────────
# 7 · Pipeline
# ─────────────────────────────────────────────────────────────────────────────

def pipeline():
    df = load_data(CSV_IN)
    df = feature_engineering(df)
    num_cols = df.select_dtypes(include="number").columns.tolist()
    df = scale_numeric(df, num_cols)
    df_pca, pca = pca_reduce(df[num_cols], var_ratio=VAR_RATIO, n_components=N_COMPONENTS)
    df_full = pd.concat([df, df_pca], axis=1)
    df_full = cluster_and_profile(df_full, df_pca.columns.tolist())

    # Görseller
    visualize_counts(df_full, "kmeans_id", "kmeans_risk", "KMeans (PCA)")
    visualize_counts(df_full, "spectral_id", "spectral_risk", "Spectral (PCA)")
    visualize_counts(df_full, "dbscan_id", "dbscan_risk", "DBSCAN (PCA)")
    visualize_counts(df_full, "optics_id", "optics_risk", "OPTICS (PCA)")

    # PCA varyans eğrisi
    plt.figure(figsize=(6, 3))
    plt.plot(np.cumsum(pca.explained_variance_ratio_)); plt.xlabel("Bileşen"); plt.ylabel("Kümülatif Varyans"); plt.title("PCA Variance")
    plt.tight_layout(); plt.show()

    save_data(df_full, CSV_OUT)

# ─────────────────────────────────────────────────────────────────────────────
# 8 · Çalıştır
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    pipeline()
