# ─────────────────────────────────────────────────────────────────────────────
# 0.  IMPORTS
# ─────────────────────────────────────────────────────────────────────────────
import warnings, pathlib, json, sys
from typing import List, Dict, Tuple

import pandas as pd
import numpy  as np
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats              import zscore
from sklearn.preprocessing    import RobustScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics          import (
    silhouette_score, calinski_harabasz_score, davies_bouldin_score
)
from sklearn.cluster          import (
    KMeans, AgglomerativeClustering, SpectralClustering, DBSCAN, OPTICS
)
from sklearn.mixture          import GaussianMixture
from sklearn.ensemble         import RandomForestClassifier
from sklearn.model_selection  import train_test_split
import shap       # açıklanabilirlik
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
    df.set_index("Lokasyon", inplace=True)
    return df


def save_data(df: pd.DataFrame, path: str) -> None:
    df.to_csv(path, index=True)
    print(f"✔  Sonuçlar «{path}» dosyasına kaydedildi.")


# ─────────────────────────────────────────────────────────────────────────────
# 1.1  VERİ ANALİZİ
# ─────────────────────────────────────────────────────────────────────────────
def analyze_data(df: pd.DataFrame) -> Dict:
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    
    # Eksik veri analizi
    missing = pd.DataFrame({
        'Missing_Count': df.isnull().sum(),
        'Missing_Percent': (df.isnull().sum() / len(df)) * 100
    }).sort_values('Missing_Percent', ascending=False)
    
    # Aykırı değer analizi (IQR yöntemi)
    outliers = {}
    for col in numeric_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        outliers[col] = len(df[(df[col] < lower) | (df[col] > upper)])
    
    # Korelasyon analizi
    correlation = df[numeric_cols].corr()
    
    return {
        "missing": missing,
        "outliers": outliers,
        "correlation": correlation
    }

def correlation_analysis(df: pd.DataFrame, x: str, y: str) -> float:
    return df[[x, y]].corr().iloc[0,1]

numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
for col1 in numeric_cols:
    for col2 in numeric_cols:
        if col1 < col2:
            corr = correlation_analysis(df, col1, col2)
            print(f"{col1} - {col2}: {corr:.3f}")



# ─────────────────────────────────────────────────────────────────────────────
# 2.  FEATURE ENGINEERING  (türetilmiş sütunlar)
# ─────────────────────────────────────────────────────────────────────────────
def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    # ---------- Demografi ----------
    df["yas_0_15_pct"]  = 100 * df["0-15 Kişi Sayısı"] / df["Toplam Nüfus"]
    df["yas_55p_pct"]   = 100 * df["55+ Kişi Sayısı"]  / df["Toplam Nüfus"]
    df["erkek_kadin_ratio"] = df["Erkek Nüfusu"] / df["Kadın Nüfusu"].replace(0, np.nan)

    # ---------- Gelir & Harcama ----------
    df["kisi_basi_gelir"] = (
        df["Aylık Ortalama Hane Geliri"] / df["Ortalama Hanehalkı"]
    )
    df["tasarruf_oran"] = (
        (df["Aylık Ortalama Hane Geliri"] - df["Aylık Ortalama Hane Harcaması"])
        / df["Aylık Ortalama Hane Geliri"]
    )
    df["borc_gelir_oran"] = (
        df["Kullanılan Toplam Kredi (BinTL)"] / df["Toplam Mevduat (Bin TL)"]
    )

    # ---------- Finansal erişim ----------
    df["kart_kullanim_oran"] = (
        (df["Banka Kartı Sayısı"] + df["Kredi Kartı Sayısı"]) / df["Toplam Nüfus"]
    )
    df["kredi_penetrasyon"] = (
        df["Kullanılan Bireysel Kredi (BinTL)"] / df["Toplam Nüfus"]
    )
    df["mevduat_kisi"] = df["Toplam Mevduat (Bin TL)"] / df["Toplam Nüfus"]

    # ---------- İstihdam ----------
    df["calisan_oran"] = df["Çalışan Oran"]        # zaten % değer
    df["beyaz_mavi_ratio"] = (
        df["Beyaz Yakalı Çalışan (Kişi)"]
        / (df["Mavi Yakalı Çalışan (Kişi)"] + 1)
    )

    # ---------- Gayrimenkul ----------
    df["konut_fiyat_gelir"] = (
        df["Ortalama Kira Değerleri"] / df["kisi_basi_gelir"].replace(0, np.nan)
    )
    df["konut_satis_hiz"] = (
        df["Toplam Konut Satış"] / df["Konut Sayısı"].replace(0, np.nan)
    )

    # ---------- Mobilite ----------
    df["yaya_trafik_log"] = np.log1p(df["Yaya Trafiği"])
    df["arac_sahip_kisi"] = df["Toplam Araç Sayısı (Adet)"] / df["Toplam Nüfus"]

    # ---------- Deprem birleşik skor ----------
    z_dep   = zscore(df["Deprem Index"])
    z_fay   = zscore(df["Fay Index"])
    z_zemin = zscore(df["Zemin Index"])
    z_tsun  = zscore(df["Tsunami Index"])
    z_volk  = zscore(df["Volkan Index"])
    z_nufus = zscore(df["Nüfus Index"])
    df["deprem_risk_score"] = (
        0.35 * z_dep   +
        0.20 * z_fay   +
        0.15 * z_zemin +
        0.10 * z_tsun  +
        0.05 * z_volk  +
        0.15 * z_nufus
    )

    # ---------- Etkileşim ----------
    df["tasarruf_x_deprem"] = df["tasarruf_oran"] * df["deprem_risk_score"]

    return df


# ─────────────────────────────────────────────────────────────────────────────
# 3.  FEATURE SEÇİMİ   (Varyans + korelasyon)
# ─────────────────────────────────────────────────────────────────────────────
def select_features(df_num: pd.DataFrame,
                    var_thresh: float = .01,
                    corr_thresh: float = .85) -> List[str]:

    vt = VarianceThreshold(threshold=var_thresh).fit(df_num)
    cols_var = df_num.columns[vt.get_support()].tolist()

    corr_mat = df_num[cols_var].corr().abs()
    upper    = corr_mat.where(np.triu(np.ones(corr_mat.shape), k=1).astype(bool))
    drop_corr= [c for c in upper.columns if any(upper[c] > corr_thresh)]
    return [c for c in cols_var if c not in drop_corr]


# ─────────────────────────────────────────────────────────────────────────────
# 4.  ÖLÇEKLEME
# ─────────────────────────────────────────────────────────────────────────────
def scale_numeric(df: pd.DataFrame, num_cols: List[str]) -> Tuple[pd.DataFrame, RobustScaler]:
    scaler = RobustScaler()
    df[num_cols] = scaler.fit_transform(df[num_cols])
    return df, scaler


# ─────────────────────────────────────────────────────────────────────────────
# 5.  OPTIMAL K
# ─────────────────────────────────────────────────────────────────────────────
def choose_k(X: pd.DataFrame, k_range=range(2, 9)) -> int:
    best_k, best_score = 2, -1
    for k in k_range:
        km = KMeans(n_clusters=k, n_init=10, random_state=42).fit(X)
        sc = silhouette_score(X, km.labels_)
        if sc > best_score:
            best_k, best_score = k, sc
    return best_k


# ─────────────────────────────────────────────────────────────────────────────
# 6.  OTOMATİK RİSK HARİTALAMA   (key_metrics parametresi yok!)
# ─────────────────────────────────────────────────────────────────────────────
def assign_risk(cluster_summary: pd.DataFrame) -> Dict[int, str]:
    """
    Ölçeklenmiş tüm özellikler kullanılarak:
      • yüzdelik rütbe -> küçük ortalama == düşük skor == High-Risk
    """
    pct_rank = cluster_summary.rank(pct=True)
    cluster_score = pct_rank.mean(axis=1)          # 0–1 arası
    order = cluster_score.sort_values().index      # düşük skor başta
    risk = {}
    if len(order) == 1:
        risk[order[0]] = "Mid-Risk"
    elif len(order) == 2:
        risk[order[0]] = "High-Risk"
        risk[order[1]] = "Low-Risk"
    else:
        risk[order[0]]      = "High-Risk"
        risk[order[-1]]     = "Low-Risk"
        for cid in order[1:-1]:
            risk[cid] = "Mid-Risk"
    return risk


# ─────────────────────────────────────────────────────────────────────────────
# 7.  CLUSTER & PROFILE   (key_metrics parametresi kalktı)
# ─────────────────────────────────────────────────────────────────────────────
def cluster_and_profile(df: pd.DataFrame, features: List[str]) -> pd.DataFrame:
    X = df[features].astype(float)
    k_opt = choose_k(X)

    # ----- K-gerektiren algoritmalar
    algos_k = {
        "kmeans":   KMeans(n_clusters=k_opt, n_init=10, random_state=42),
        "spectral": SpectralClustering(n_clusters=k_opt, assign_labels="kmeans",
                                       random_state=42),
        "gmm":      GaussianMixture(n_components=k_opt, random_state=42),
        "agg":      AgglomerativeClustering(n_clusters=k_opt, linkage="ward"),
    }
    for name, model in algos_k.items():
        df[f"{name}_id"] = model.fit_predict(X)
        summary = df.groupby(f"{name}_id")[features].median()
        df[f"{name}_risk"] = df[f"{name}_id"].map(assign_risk(summary))

    # ----- Yoğunluk tabanlı algoritmalar
    density = {
        "dbscan": DBSCAN(eps=0.8, min_samples=10),
        "optics": OPTICS(min_samples=10, xi=.05, min_cluster_size=.02),
    }
    for name, model in density.items():
        labels = model.fit_predict(X)
        df[f"{name}_id"] = labels
        mask = labels != -1
        if mask.sum():                                  # en az bir küme varsa
            summary = df.loc[mask].groupby(f"{name}_id")[features].median()
            mapper  = assign_risk(summary)
            df[f"{name}_risk"] = df[f"{name}_id"].map(mapper)
            df.loc[~mask, f"{name}_risk"] = "Noise"
        else:
            df[f"{name}_risk"] = "Noise"
    return df

# ─────────────────────────────────────────────────────────────────────────────
# 8.  SHAP EXPLAINABILITY
# ─────────────────────────────────────────────────────────────────────────────
def shap_explain(df: pd.DataFrame, features: List[str], target_col: str, max_display: int = 15):
    X = df[features]
    y = df[target_col].map({"Low-Risk": 0, "Mid-Risk": 1, "High-Risk": 2})
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=.3, stratify=y, random_state=42
    )
    rf = RandomForestClassifier(n_estimators=400, random_state=42).fit(X_train, y_train)
    explainer = shap.TreeExplainer(rf)
    shap_vals = explainer.shap_values(X_val, check_additivity=False)
    shap.summary_plot(shap_vals, X_val, show=False,
                      plot_type="bar", max_display=max_display)
    plt.title("SHAP Feature Importance (RF surrogate)"); plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# 9.  GÖRSEL YARDIMCI
# ─────────────────────────────────────────────────────────────────────────────
def visualize_counts(df: pd.DataFrame, cluster_col: str, risk_col: str, title: str):
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    sns.countplot(x=df[cluster_col], ax=ax[0]); ax[0].set_title(f"{title} Clusters")
    sns.countplot(x=df[risk_col],
                  order=["Low-Risk", "Mid-Risk", "High-Risk", "Noise"],
                  ax=ax[1])
    ax[1].set_title(f"{title} Risk"); plt.tight_layout(); plt.show()


def main(csv_in: str = "data.csv", csv_out: str = "risk_segmented.csv"):
    df        = load_data(csv_in)                   # 1
    df        = feature_engineering(df)             # 2  (kendi türettiğiniz tüm ek değişkenler dahildir)
    num_cols  = df.select_dtypes(include="number").columns
    selected  = select_features(df[num_cols])       # 3  (istatistiksel filtreler)
    df, _     = scale_numeric(df, selected)         # 4

    df        = cluster_and_profile(df, selected)   # 5-7 (artık key_metrics yok)

    # Örnek görselleştirme (opsiyonel)
    visualize_counts(df, "kmeans_id",   "kmeans_risk",   "KMeans")
    visualize_counts(df, "spectral_id", "spectral_risk", "Spectral")
    visualize_counts(df, "dbscan_id",   "dbscan_risk",   "DBSCAN")
    visualize_counts(df, "optics_id",   "optics_risk",   "OPTICS")

    shap_explain(df, selected, "kmeans_risk")       # 8
    save_data(df, csv_out)             