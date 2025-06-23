import pandas as pd
import matplotlib.pyplot as plt
from ace_tools import display_dataframe_to_user

def eda_report(df: pd.DataFrame, target: str = None, plots: bool = True):
    """
    Comprehensive EDA report including:
      1. Basic info & missing values
      2. Full dataset display
      3. Numeric descriptive statistics
      4. Missing value analysis
      5. Histograms & boxplots for numeric features (optional)
      6. Value counts & bar plots for categorical features (optional)
      7. If target provided:
         - Categorical vs target relationship (tables + optional plots)
         - Numeric vs target relationship (tables + optional plots)
         - Correlation with target (tables + optional plots)
    
    Args:
        df (pd.DataFrame): Input DataFrame
        target (str): Name of target column. If None, target-related analyses are skipped.
        plots (bool): Whether to show plots (True) or skip them (False).
    """
    # 1. Basic Info & Missing
    info = pd.DataFrame({
        'dtype': df.dtypes,
        'missing': df.isnull().sum(),
        'missing_pct': (df.isnull().mean() * 100).round(2),
        'unique': df.nunique()
    }).sort_values('missing', ascending=False)
    display_dataframe_to_user("Basic Info & Missing", info)

    # 2. Full Data Display
    display_dataframe_to_user("Full Dataset", df)

    # 3. Numeric Descriptive Statistics
    num_df = df.select_dtypes(include='number')
    desc = num_df.describe().T
    display_dataframe_to_user("Numeric Descriptive Statistics", desc)

    # 4. Missing Value Analysis
    missing = pd.DataFrame({
        'missing': df.isnull().sum(),
        'missing_pct': (df.isnull().mean() * 100).round(2)
    }).loc[lambda x: x['missing'] > 0].sort_values('missing', ascending=False)
    display_dataframe_to_user("Missing Value Analysis", missing)

    # 5. Numeric Plots
    if plots:
        for col in num_df.columns:
            plt.figure()
            plt.hist(df[col].dropna(), bins=30)
            plt.title(f"Histogram of {col}")
            plt.xlabel(col)
            plt.ylabel("Frequency")
            plt.show()

            plt.figure()
            plt.boxplot(df[col].dropna(), vert=False)
            plt.title(f"Boxplot of {col}")
            plt.xlabel(col)
            plt.show()

    # 6. Categorical Value Counts & Optional Bar Plots
    cat_cols = df.select_dtypes(include=['object', 'category']).columns
    for col in cat_cols:
        counts = df[col].value_counts()
        display_dataframe_to_user(f"Value Counts for {col}", counts.to_frame(name='count'))
        if plots:
            plt.figure(figsize=(8, 4))
            plt.bar(counts.index.astype(str), counts.values)
            plt.xticks(rotation=45, ha='right')
            plt.title(f"Value Counts - {col}")
            plt.xlabel(col)
            plt.ylabel("Count")
            plt.tight_layout()
            plt.show()

    # 7. Target-based analyses
    if target and target in df.columns:
        # Categorical vs Target
        for col in cat_cols:
            if col == target:
                continue
            ct = df.groupby(col)[target].agg(['count', 'mean']).rename(columns={'mean':'target_rate'})
            display_dataframe_to_user(f"Target by {col}", ct)
            if plots:
                plt.figure(figsize=(8, 4))
                plt.bar(ct.index.astype(str), ct['target_rate'])
                plt.xticks(rotation=45, ha='right')
                plt.title(f"Mean {target} by {col}")
                plt.ylabel(f"Mean {target}")
                plt.tight_layout()
                plt.show()

        # Numeric vs Target
        for col in num_df.columns:
            if col == target:
                continue
            stats = df.groupby(target)[col].agg(['mean', 'median', 'std', 'count'])
            display_dataframe_to_user(f"{col} stats by {target}", stats)
            if plots:
                plt.figure()
                df.boxplot(column=col, by=target, grid=False)
                plt.title(f"{col} distribution by {target}")
                plt.suptitle("")
                plt.show()

        # Correlation with Target
        feats = [c for c in df.columns if c != target and c in num_df.columns]
        corr_target = df[feats + [target]].corr()[target].drop(target).sort_values(key=abs, ascending=False)
        display_dataframe_to_user("Correlation with Target", corr_target.to_frame(name='corr_with_target'))
        if plots:
            plt.figure(figsize=(8, 4))
            plt.bar(corr_target.index, corr_target.values)
            plt.xticks(rotation=90)
            plt.title(f"Feature Correlation with {target}")
            plt.tight_layout()
            plt.show()

# Usage example:
# df = pd.read_csv("veri.csv")
# eda_report(df, target='hedef', plots=True)
