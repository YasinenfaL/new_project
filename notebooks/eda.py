"""
Keşifsel Veri Analizi (EDA).
Veri setlerinin detaylı analizini gerçekleştiren kodlar içerir.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sqlalchemy
from scipy.stats import chi2_contingency

def load_data(connection_string: str, query: str) -> pd.DataFrame:
    """Veriyi SQL veritabanından yükler ve DataFrame olarak döndürür."""
    try:
        engine = sqlalchemy.create_engine(connection_string)
        df = pd.read_sql(query, engine)
        engine.dispose()
        return df
    except Exception as e:
        print(f"Veri yükleme hatası: {e}")
        return None

def analyze_data_structure(df: pd.DataFrame) -> None:
    """Veri setinin genel yapısını analiz eder."""
    print("=" * 50)
    print("VERİ SETİ YAPISI")
    print("=" * 50)
    print(f"Satır sayısı: {df.shape[0]}")
    print(f"Sütun sayısı: {df.shape[1]}")
    print("\nDeğişken tipleri:")
    print(df.dtypes)
    print("\nBellek kullanımı:")
    print(df.memory_usage(deep=True))

def display_head(df: pd.DataFrame, n: int = 5) -> None:
    """Veri setinin ilk n satırını gösterir."""
    print("=" * 50)
    print(f"İLK {n} SATIR")
    print("=" * 50)
    print(df.head(n))

def analyze_missing_values(df: pd.DataFrame) -> None:
    """Eksik değerleri analiz eder ve görselleştirir."""
    print("=" * 50)
    print("EKSİK DEĞER ANALİZİ")
    print("=" * 50)
    
    missing_summary = pd.DataFrame({
        'Eksik Değer Sayısı': df.isnull().sum(),
        'Eksik Değer Oranı (%)': (df.isnull().sum() / len(df)) * 100
    }).sort_values('Eksik Değer Sayısı', ascending=False)
    
    print("\nEksik Değer Özeti:")
    print(missing_summary)
    
    # Eksik değer görselleştirmeleri
    _, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    sns.heatmap(df.isnull(), yticklabels=False, cbar=False, cmap='viridis', ax=ax1)
    ax1.set_title('Eksik Değer Haritası')
    
    missing_summary['Eksik Değer Sayısı'].plot(kind='bar', ax=ax2)
    ax2.set_title('Değişkenlerdeki Eksik Değer Sayıları')
    ax2.set_xlabel('Değişkenler')
    ax2.set_ylabel('Eksik Değer Sayısı')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def identify_numeric_types(df: pd.DataFrame) -> tuple:
    """Numerik değişkenleri kesikli ve sürekli olarak ayırır."""
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
    discrete_cols = []
    continuous_cols = []
    
    for col in numeric_cols:
        unique_ratio = len(df[col].unique()) / len(df)
        if unique_ratio < 0.05:  # Eşik değeri
            discrete_cols.append(col)
        else:
            continuous_cols.append(col)
    
    return discrete_cols, continuous_cols

def analyze_discrete_variables(df: pd.DataFrame, discrete_cols: list) -> None:
    """Kesikli değişkenler için görselleştirmeler yapar."""
    print("=" * 50)
    print("KESİKLİ DEĞİŞKEN ANALİZİ")
    print("=" * 50)
    
    if not discrete_cols:
        print("Kesikli değişken bulunamadı.")
        return
    
    for col in discrete_cols:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        sns.countplot(data=df, x=col, ax=ax1)
        ax1.set_title(f'{col} - Değer Dağılımı')
        ax1.set_xlabel(col)
        ax1.set_ylabel('Frekans')
        ax1.tick_params(axis='x', rotation=45)
        
        sns.boxplot(data=df, y=col, ax=ax2)
        ax2.set_title(f'{col} - Kutu Grafiği')
        
        plt.tight_layout()
        plt.show()
        
        print(f"\n{col} Özet İstatistikleri:")
        print(df[col].describe())

def analyze_continuous_variables(df: pd.DataFrame, continuous_cols: list) -> None:
    """Sürekli değişkenler için görselleştirmeler yapar."""
    print("=" * 50)
    print("SÜREKLİ DEĞİŞKEN ANALİZİ")
    print("=" * 50)
    
    if not continuous_cols:
        print("Sürekli değişken bulunamadı.")
        return
    
    for col in continuous_cols:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        sns.histplot(data=df, x=col, kde=True, ax=ax1)
        ax1.set_title(f'{col} - Dağılım Grafiği')
        ax1.set_xlabel(col)
        ax1.set_ylabel('Frekans')
        
        sns.boxplot(data=df, y=col, ax=ax2)
        ax2.set_title(f'{col} - Kutu Grafiği')
        
        plt.tight_layout()
        plt.show()
        
        print(f"\n{col} Özet İstatistikleri:")
        print(df[col].describe())
        print(f"Çarpıklık: {df[col].skew():.2f}")
        print(f"Basıklık: {df[col].kurtosis():.2f}")

def identify_categorical_columns(df: pd.DataFrame) -> list:
    """Kategorik değişkenleri tespit eder."""
    cat_cols = df.select_dtypes(include=['object', 'category']).columns
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
    for col in numeric_cols:
        if len(df[col].unique()) < 10:  # Eşik değeri
            cat_cols = cat_cols.append(pd.Index([col]))
    
    return list(cat_cols)

def analyze_categorical_variables(df: pd.DataFrame, cat_cols: list) -> None:
    """Kategorik değişkenler için görselleştirmeler yapar."""
    print("=" * 50)
    print("KATEGORİK DEĞİŞKEN ANALİZİ")
    print("=" * 50)
    
    if not cat_cols:
        print("Kategorik değişken bulunamadı.")
        return
    
    for col in cat_cols:
        value_counts = df[col].value_counts()
        value_percentages = df[col].value_counts(normalize=True) * 100
        
        plt.figure(figsize=(min(15, max(8, len(value_counts)/2)), 6))
        ax = sns.barplot(x=value_counts.index, y=value_counts.values)
        
        for i, (count, percentage) in enumerate(zip(value_counts, value_percentages)):
            ax.text(i, count, f'{count}\n({percentage:.1f}%)', ha='center', va='bottom')
        
        plt.title(f'{col} Değişkeni Dağılımı')
        plt.xlabel(col)
        plt.ylabel('Frekans')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.show()
        
        print(f"\n{col} Değişkeni İstatistikleri:")
        summary_df = pd.DataFrame({
            'Değer': value_counts.index,
            'Frekans': value_counts.values,
            'Yüzde (%)': value_percentages
        })
        print(summary_df)
        print(f"\nBenzersiz Değer Sayısı: {df[col].nunique()}")
        print(f"En Sık Görülen Değer: {df[col].mode().values[0]}")
        print("=" * 50)

def analyze_missing_by_target(df: pd.DataFrame, target_col: str) -> None:
    """Eksik değerlerin hedef değişkene göre dağılımını analiz eder."""
    print("=" * 50)
    print("HEDEF DEĞİŞKENE GÖRE EKSİK DEĞER ANALİZİ")
    print("=" * 50)
    
    target_dist = df[target_col].value_counts(normalize=True) * 100
    print("\nHedef Değişken Dağılımı:")
    for cls, pct in target_dist.items():
        print(f"Sınıf {cls}: %{pct:.2f}")
    
    missing_cols = df.columns[df.isnull().any()].tolist()
    
    if not missing_cols:
        print("\nVeri setinde eksik değer bulunmamaktadır.")
        return
    
    for col in missing_cols:
        if col != target_col:
            print(f"\n{'-'*20}\nDeğişken: {col}")
            
            missing_target_dist = df[df[col].isnull()][target_col].value_counts(normalize=True) * 100
            non_missing_target_dist = df[df[col].notnull()][target_col].value_counts(normalize=True) * 100
            
            comparison_df = pd.DataFrame({
                'Genel Dağılım (%)': target_dist,
                'Eksik Değerlerde (%)': missing_target_dist,
                'Dolu Değerlerde (%)': non_missing_target_dist
            }).round(2)
            
            print("\nHedef Değişken Dağılımı Karşılaştırması:")
            print(comparison_df)
            
            observed = pd.crosstab(df[col].isnull(), df[target_col])
            _, p_value, _, _ = chi2_contingency(observed)
            print(f"\nKi-kare testi p-değeri: {p_value:.4f}")
            
            plt.figure(figsize=(12, 5))
            
            plt.subplot(1, 2, 1)
            df[df[col].isnull()][target_col].value_counts().plot(kind='pie', 
                autopct='%1.1f%%', title=f'Eksik Değerlerde\nHedef Dağılımı')
            
            plt.subplot(1, 2, 2)
            df[df[col].notnull()][target_col].value_counts().plot(kind='pie', 
                autopct='%1.1f%%', title=f'Dolu Değerlerde\nHedef Dağılımı')
            
            plt.suptitle(f'{col} Değişkeni için Hedef Dağılımı Karşılaştırması')
            plt.tight_layout()
            plt.show()
            
            print("\nDengesizlik Analizi:")
            for cls in df[target_col].unique():
                total_class = (df[target_col] == cls).sum()
                missing_class = df[(df[target_col] == cls) & (df[col].isnull())].shape[0]
                missing_rate = (missing_class / total_class) * 100
                
                print(f"Sınıf {cls}:")
                print(f"  Toplam örnek sayısı: {total_class}")
                print(f"  Eksik değer sayısı: {missing_class}")
                print(f"  Eksik değer oranı: %{missing_rate:.2f}")

def detect_outliers(df: pd.DataFrame, column: str) -> pd.Index:
    """Aykırı değerleri IQR yöntemi ile tespit eder."""
    q1 = df[column].quantile(0.25)
    q3 = df[column].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    return df[(df[column] < lower_bound) | (df[column] > upper_bound)].index

def analyze_outliers(df: pd.DataFrame, continuous_cols: list, target_col: str) -> None:
    """Aykırı değerleri analiz eder ve hedef değişken sınıflarına göre dağılımını inceler."""
    print("=" * 50)
    print("AYKIRI DEĞER ANALİZİ")
    print("=" * 50)
    
    for col in continuous_cols:
        outliers = detect_outliers(df, col)
        num_outliers = len(outliers)
        print(f"\nDeğişken: {col}")
        print(f"Aykırı Değer Sayısı: {num_outliers}")
        
        if num_outliers > 0:
            outlier_target_dist = df.loc[outliers, target_col].value_counts(normalize=True) * 100
            non_outlier_target_dist = df.loc[~df.index.isin(outliers), target_col].value_counts(normalize=True) * 100
            
            comparison_df = pd.DataFrame({
                'Genel Dağılım (%)': df[target_col].value_counts(normalize=True) * 100,
                'Aykırı Değerlerde (%)': outlier_target_dist,
                'Aykırı Olmayanlarda (%)': non_outlier_target_dist
            }).round(2)
            
            print("\nHedef Değişken Dağılımı Karşılaştırması:")
            print(comparison_df)
            
            plt.figure(figsize=(12, 5))
            
            plt.subplot(1, 2, 1)
            df.loc[outliers, target_col].value_counts().plot(kind='pie', 
                autopct='%1.1f%%', title=f'Aykırı Değerlerde\nHedef Dağılımı')
            
            plt.subplot(1, 2, 2)
            df.loc[~df.index.isin(outliers), target_col].value_counts().plot(kind='pie', 
                autopct='%1.1f%%', title=f'Aykırı Olmayanlarda\nHedef Dağılımı')
            
            plt.suptitle(f'{col} Değişkeni için Hedef Dağılımı Karşılaştırması')
            plt.tight_layout()
            plt.show()

def analyze_rare_categories(df: pd.DataFrame, cat_cols: list, target_col: str, rare_threshold: float = 0.01) -> None:
    """Kategorik değişkenlerdeki nadir sınıfları analiz eder."""
    print("=" * 50)
    print("NADİR KATEGORİ ANALİZİ")
    print("=" * 50)
    
    for col in cat_cols:
        if col != target_col:
            print(f"\n{'-'*20}\nDeğişken: {col}")
            
            value_counts = df[col].value_counts()
            value_percentages = df[col].value_counts(normalize=True)
            
            rare_categories = value_percentages[value_percentages < rare_threshold].index
            
            if not rare_categories:
                print(f"Bu değişkende nadir kategori bulunmamaktadır. (Eşik: %{rare_threshold*100:.1f})")
                continue
                
            print(f"\nNadir Kategoriler (<%{rare_threshold*100:.1f}):")
            rare_summary = pd.DataFrame({
                'Kategori': rare_categories,
                'Frekans': value_counts[rare_categories],
                'Yüzde (%)': value_percentages[rare_categories] * 100
            })
            print(rare_summary)
            
            print("\nNadir Kategorilerin Hedef Değişken Dağılımı:")
            rare_target_dist = df[df[col].isin(rare_categories)][target_col].value_counts(normalize=True) * 100
            non_rare_target_dist = df[~df[col].isin(rare_categories)][target_col].value_counts(normalize=True) * 100
            
            comparison_df = pd.DataFrame({
                'Genel Dağılım (%)': df[target_col].value_counts(normalize=True) * 100,
                'Nadir Kategorilerde (%)': rare_target_dist,
                'Diğer Kategorilerde (%)': non_rare_target_dist
            }).round(2)
            
            print(comparison_df)
            
            plt.figure(figsize=(15, 10))
            
            plt.subplot(2, 1, 1)
            sns.barplot(x=rare_categories, y=value_counts[rare_categories])
            plt.title(f'Nadir Kategorilerin Frekans Dağılımı - {col}')
            plt.xticks(rotation=45, ha='right')
            
            plt.subplot(2, 1, 2)
            
            target_props = []
            categories = []
            
            for category in rare_categories:
                category_data = df[df[col] == category][target_col].value_counts(normalize=True) * 100
                target_props.append(category_data)
                categories.append(category)
            
            target_prop_df = pd.DataFrame(target_props, index=categories)
            
            target_prop_df.plot(kind='bar', stacked=True)
            plt.title(f'Nadir Kategorilerde Hedef Değişken Dağılımı - {col}')
            plt.xlabel('Kategoriler')
            plt.ylabel('Yüzde (%)')
            plt.legend(title='Hedef Sınıflar', bbox_to_anchor=(1.05, 1))
            plt.xticks(rotation=45, ha='right')
            
            plt.tight_layout()
            plt.show()
            
            contingency_table = pd.crosstab(df[col].isin(rare_categories), df[target_col])
            _, p_value, _, _ = chi2_contingency(contingency_table)
            print(f"\nKi-kare testi p-değeri: {p_value:.4f}")
            
            print("\nNadir Kategorilerin Etki Analizi:")
            total_rare = df[col].isin(rare_categories).sum()
            total_samples = len(df)
            print(f"Toplam nadir kategori örneği: {total_rare} (%{(total_rare/total_samples)*100:.2f})")
            
            for target_class in df[target_col].unique():
                rare_class = df[(df[col].isin(rare_categories)) & (df[target_col] == target_class)].shape[0]
                total_class = df[target_col].value_counts()[target_class]
                print(f"\nHedef Sınıf {target_class}:")
                print(f"  Nadir kategorilerdeki örnek sayısı: {rare_class}")
                print(f"  Sınıftaki toplam örnek sayısı: {total_class}")
                print(f"  Nadir kategori oranı: %{(rare_class/total_class)*100:.2f}")

def analyze_correlations(df: pd.DataFrame, numeric_cols: list, target_col: str = None) -> None:
    """Numerik değişkenler arasındaki korelasyonları analiz eder."""
    print("=" * 50)
    print("KORELASYON ANALİZİ")
    print("=" * 50)
    
    if len(numeric_cols) < 2:
        print("Korelasyon analizi için en az 2 numerik değişken gereklidir.")
        return
    
    corr_matrix = df[numeric_cols].corr()
    
    plt.figure(figsize=(12, 8))
    sns.heatmap(corr_matrix, 
                annot=True,
                cmap='coolwarm',
                center=0,
                fmt='.2f',
                square=True,
                mask=np.triu(np.ones_like(corr_matrix, dtype=bool)))
    plt.title('Değişkenler Arası Korelasyon Matrisi')
    plt.tight_layout()
    plt.show()
    
    print("\nYüksek Korelasyonlu Değişken Çiftleri (|r| > 0.7):")
    high_corr_pairs = []
    
    for i, col1 in enumerate(numeric_cols):
        for j, col2 in enumerate(numeric_cols[i+1:], i+1):
            corr = corr_matrix.iloc[i, j]
            if abs(corr) > 0.7:
                high_corr_pairs.append({
                    'Değişken 1': col1,
                    'Değişken 2': col2,
                    'Korelasyon': corr
                })
    
    if high_corr_pairs:
        high_corr_df = pd.DataFrame(high_corr_pairs)
        print(high_corr_df.sort_values('Korelasyon', key=abs, ascending=False))
        
        print("\nYüksek Korelasyonlu Değişkenlerin Dağılım Grafikleri:")
        for pair in high_corr_pairs:
            plt.figure(figsize=(8, 6))
            sns.scatterplot(data=df, x=pair['Değişken 1'], y=pair['Değişken 2'])
            plt.title(f"Korelasyon: {pair['Korelasyon']:.2f}")
            plt.tight_layout()
            plt.show()
    else:
        print("Yüksek korelasyonlu değişken çifti bulunamadı.")
    
    if target_col is not None and len(df[target_col].unique()) == 2:
        print("\nHedef Değişken ile Point-Biserial Korelasyonlar:")
        
        target_correlations = {}
        for col in numeric_cols:
            correlation = df[col].corr(df[target_col])
            target_correlations[col] = correlation
        
        target_correlations = pd.Series(target_correlations).sort_values(key=abs, ascending=False)
        print("\nHedef Değişken ile Korelasyonlar (mutlak değere göre sıralı):")
        for col, corr in target_correlations.items():
            print(f"{col}: {corr:.3f}")
        
        plt.figure(figsize=(12, 6))
        target_correlations.plot(kind='bar')
        plt.title('Değişkenlerin Hedef Değişken ile Point-Biserial Korelasyonu')
        plt.xlabel('Değişkenler')
        plt.ylabel('Korelasyon Katsayısı')
        plt.xticks(rotation=45, ha='right')
        plt.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        plt.tight_layout()
        plt.show()
        
        print("\nEn Güçlü Korelasyonlu Değişkenler için Box Plot:")
        top_corr_features = target_correlations.head(3).index
        
        for feature in top_corr_features:
            plt.figure(figsize=(8, 6))
            sns.boxplot(data=df, x=target_col, y=feature)
            plt.title(f'{feature} - Hedef Değişken İlişkisi\nKorelasyon: {target_correlations[feature]:.3f}')
            plt.tight_layout()
            plt.show()

def main(data_path: str, target_col: str = None) -> None:
    """Ana fonksiyon: Tüm analizleri sırayla gerçekleştirir."""
    df = load_data(data_path)
    if df is None:
        return
    
    analyze_data_structure(df)
    display_head(df)
    analyze_missing_values(df)
    
    if target_col is not None and target_col in df.columns:
        analyze_missing_by_target(df, target_col)
    
    discrete_cols, continuous_cols = identify_numeric_types(df)
    analyze_discrete_variables(df, discrete_cols)
    analyze_continuous_variables(df, continuous_cols)
    
    if target_col is not None:
        numeric_cols = discrete_cols + continuous_cols
        analyze_correlations(df, numeric_cols, target_col)

# Sabitler
FILE_PATH = "your_data.csv"
TARGET_COL = "target"

if __name__ == "__main__":
    main(FILE_PATH, TARGET_COL)
