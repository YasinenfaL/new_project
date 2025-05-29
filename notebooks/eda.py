#!/usr/bin/env python

-- coding: utf-8 --

""" Risk Segmentation Pipeline – Log-Multiplicative Risk Score

1. CSV’den lokasyon bazlı özellikleri yükler


2. Gruplara göre özellikleri MinMax ölçekler ve yön düzeltmesi yapar


3. Feature impact simülasyonu ile her değişkenin ağırlığını belirler


4. Risk skoru = 100 * exp(∑ w_i * ln(x_i + EPS)) → [0-100] aralığına MinMax normalize


5. Elbow yöntemi (KMeans SSE) ile optimal küme sayısını bulur


6. Risk skoru üzerinden segmentasyon yapar, sonuçları CSV’ye yazar



Kullanım: python risk_segmentation_log.py --csv_in data.csv --csv_out results.csv --max_k 10 """ import argparse import pandas as pd import numpy as np from sklearn.preprocessing import MinMaxScaler from sklearn.cluster import KMeans

-------------------------------------------------------------------------

1. Veri Yükleme

-------------------------------------------------------------------------

def load_data(path: str) -> pd.DataFrame: df = pd.read_csv(path) if 'Lokasyon' in df.columns: df = df.set_index('Lokasyon') return df

-------------------------------------------------------------------------

2. Ölçekleme & Yön Düzeltmesi

-------------------------------------------------------------------------

GROUPS = { "Gelir & Harcama": {"features": ["Aylık Ortalama Hane Geliri","tasarruf_oran","Bireysel Kredi / Gelir","Toplam Nevduat / Gelir","gelir_kira_yuku","Toplam Harcama","Max_Min_Income_Ratio","Income_Gini_Proxy"]}, "Kredi & Bankacılık": {"features":["Kullanılan Toplam Kredi (BinTL)","kart_kisi_oran"]}, "Coğrafi & Afet": {"features":["Deprem Puan","Bölge Riski"]}, "Eğitim": {"features":["Ortalama Eğitim Süresi (Yıl)","lisansüstü_oran","üniversite_oran","ilkokul_oran","ilköğretim_oran","okumamış_oran","okuryazar_oran"]}, "Tüketim & SES": {"features":["Tüketim Potansiyeli (Yüzde %)","Ortalama SES","Gelişmişlik Katsayısı","AB Oran","DF Oran"]}, "Demografi": {"features":["Ortalama Hanehalkı","çocuk_oran","genç_oran","orta_yas_oran","yaslı_oran","bekar_oran","evli_oran","Working_Age_Share","Dependency_Ratio"]}, "Çalışma": {"features":["Çalışan Oranı"]}, "Altyapı": {"features":["Konut Yoğunluğu (Konut/Km2)","Ortalama Kira Değerleri","Alan Büyüklüğü / km2","Population_Density"]}, "Varlık": {"features":["Hane Başına Araç Sahipliği"]} }

'neg' yönlü protective özellikler

NEGATIVE = {f for f in ["Aylık Ortalama Hane Geliri","Ortalama Eğitim Süresi (Yıl)","lisansüstü_oran","üniversite_oran","ilkokul_oran","ilköğretim_oran","okuryazar_oran","Ortalama SES","Gelişmişlik Katsayısı","evli_oran","Working_Age_Share","Çalışan Oranı"]}

EPS = 1e-6

def scale_and_direct(df: pd.DataFrame) -> (pd.DataFrame, list): features = [f for grp in GROUPS.values() for f in grp['features'] if f in df.columns] scaler = MinMaxScaler() df[features] = scaler.fit_transform(df[features]) for f in features: if f in NEGATIVE: df[f] = 1.0 - df[f] return df, features

-------------------------------------------------------------------------

3. Feature Impact Tabanlı Ağırlık Hesabı

-------------------------------------------------------------------------

def compute_feature_impact_weights(df: pd.DataFrame, features: list) -> dict: impacts = {} for f in features: df1 = df.copy(); df0 = df.copy() df1[f] = 1.0 df0[f] = 0.0 base1 = df1[features].sum(axis=1) base0 = df0[features].sum(axis=1) impacts[f] = (base1 - base0).mean() total = sum(impacts.values()) return {f: impacts[f]/total for f in features}

-------------------------------------------------------------------------

4. Log-Multiplicative Risk Skoru Hesaplama

-------------------------------------------------------------------------

def compute_risk_score(df: pd.DataFrame, weights: dict) -> pd.Series: ln_part = np.zeros(len(df)) for f, w in weights.items(): ln_part += w * np.log(df[f] + EPS) raw = np.exp(ln_part) # [0,100] aralığına MinMax normalize rmin, rmax = raw.min(), raw.max() return 100 * (raw - rmin) / (rmax - rmin)

-------------------------------------------------------------------------

5. Elbow Method (Optimal K)

-------------------------------------------------------------------------

def compute_sse(scores: pd.Series, max_k: int) -> (list, list, int): arr = scores.values.reshape(-1,1) ks, sse = [], [] for k in range(1, max_k+1): km = KMeans(n_clusters=k, random_state=42).fit(arr) ks.append(k); sse.append(km.inertia_) # Elastik line üzerindeki en uzak nokta p1, p2 = np.array([ks[0],sse[0]]), np.array([ks[-1],sse[-1]]) line = p2 - p1; norm = line / np.linalg.norm(line) dists = [abs(np.cross(norm, np.array([k,s]) - p1)) for k,s in zip(ks,sse)] return ks, sse, ks[int(np.argmax(dists))]

-------------------------------------------------------------------------

6. Pipeline

-------------------------------------------------------------------------

def pipeline(csv_in: str, csv_out: str, max_k: int): df = load_data(csv_in) df_scaled, features = scale_and_direct(df) weights = compute_feature_impact_weights(df_scaled, features) df['RiskScore'] = compute_risk_score(df_scaled, weights) ks, sse, optimal_k = compute_sse(df['RiskScore'], max_k) print(f"Optimal segment sayısı (elbow): {optimal_k}") km = KMeans(n_clusters=optimal_k, random_state=42).fit(df['RiskScore'].values.reshape(-1,1)) df['Segment'] = km.labels_ + 1 df.to_csv(csv_out) print(f"Sonuçlar kaydedildi: {csv_out}")

if name == 'main': ap = argparse.ArgumentParser() ap.add_argument('--csv_in', required=True) ap.add_argument('--csv_out', required=True) ap.add_argument('--max_k', type=int, default=10) args = ap.parse_args() pipeline(args.csv_in, args.csv_out, args.max_k)

