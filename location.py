from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import (
    KMeans, SpectralClustering, AgglomerativeClustering, DBSCAN, OPTICS
)
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.preprocessing import RobustScaler
from sklearn.manifold import TSNE

import warnings
warnings.filterwarnings('ignore')

import warnings
warnings.filterwarnings('ignore')
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=RuntimeWarning)


# ─────────────────────────────────────────────────────────────────────────────
# Sabitler – ihtiyaca göre güncelle
# ─────────────────────────────────────────────────────────────────────────────
CSV_IN: str = "data.csv"              # Girdi CSV yolunu ayarla
CSV_OUT: str = "risk_segmented.csv"   # Çıktı CSV yolunu ayarla
N_CLUSTERS = 4      # Küme sayısı (PCA kaldırıldı, doğrudan özelliklerle)
PERMUTE_ROUNDS = 10 # Her özellik için permutation tekrarı


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

# ─────────────────────────────────────────────────────────────────────────────
# Ölçekleme
# ─────────────────────────────────────────────────────────────────────────────
def scale_numeric(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    scaler = RobustScaler()
    df[cols] = scaler.fit_transform(df[cols])
    return df

# ─────────────────────────────────────────────────────────────────────────────
# Risk Ataması
# ─────────────────────────────────────────────────────────────────────────────
def assign_risk(df: pd.DataFrame, cluster_col: str, features: list[str]) -> pd.Series:
    summary = df.groupby(cluster_col)[features].median().mean(axis=1)
    order = summary.sort_values().index
    risk_labels = ["High-Risk", "Mid-Risk", "Low-Risk"]
    risk_map = {cluster: risk_labels[min(i, 2)] for i, cluster in enumerate(order)}
    return df[cluster_col].map(risk_map)

# ─────────────────────────────────────────────────────────────────────────────
# t-SNE Görselleştirme
# ─────────────────────────────────────────────────────────────────────────────
def plot_tsne(X: pd.DataFrame, labels, title: str):
    tsne = TSNE(n_components=2, random_state=42)
    X_emb = tsne.fit_transform(X)
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=X_emb[:, 0], y=X_emb[:, 1], hue=labels, palette='tab10')
    plt.title(title)
    plt.show()

# ─────────────────────────────────────────────────────────────────────────────
# Kümeleme & Analiz
# ─────────────────────────────────────────────────────────────────────────────
def cluster_and_analyze(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    X = df[features].astype(float)

    algos = {
        "KMeans": KMeans(n_clusters=N_CLUSTERS, random_state=42),
        "Spectral": SpectralClustering(n_clusters=N_CLUSTERS, random_state=42),
        "Agglomerative": AgglomerativeClustering(n_clusters=N_CLUSTERS),
        "DBSCAN": DBSCAN(eps=0.8, min_samples=10),
        "OPTICS": OPTICS(min_samples=10)
    }

    for name, model in algos.items():
        labels = model.fit_predict(X)
        df[f"{name}_id"] = labels

        if len(set(labels)) > 1:
            sil = silhouette_score(X, labels)
            calinski = calinski_harabasz_score(X, labels)
            davies = davies_bouldin_score(X, labels)
            print(f"{name} | Silhouette: {sil:.3f}, Calinski: {calinski:.1f}, Davies-Bouldin: {davies:.3f}")
        else:
            print(f"{name}: Tek küme bulundu.")

        df[f"{name}_risk"] = assign_risk(df, f"{name}_id", features)

        plot_tsne(X, labels, title=f"{name} t-SNE Görselleştirme")

    return df

# ─────────────────────────────────────────────────────────────────────────────
# Pipeline
# ─────────────────────────────────────────────────────────────────────────────
def pipeline():
    df = load_data(CSV_IN)
    df = feature_engineering(df)
    num_cols = df.select_dtypes(include="number").columns.tolist()
    df = scale_numeric(df, num_cols)
    df = cluster_and_analyze(df, num_cols)
    save_data(df, CSV_OUT)

# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    pipeline()
