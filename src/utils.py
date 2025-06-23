import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""  # CUDA’yı devre dışı bırak

import shap
import matplotlib.pyplot as plt

def shap_feature_analysis(model, X_train, X_test, feature_names=None, max_display=20, nsamples=100):
    """
    Model ve veri verildiğinde detaylı SHAP analizi yapar (CPU modunda).
    
    Args:
        model: Eğitilmiş model nesnesi.
        X_train: Model eğitim verisi (DataFrame veya numpy array).
        X_test: Model test verisi (DataFrame veya numpy array).
        feature_names: Özellik isimleri listesi. None ise X_train kolon isimleri kullanılır.
        max_display: Summary plot’ta gösterilecek maksimum feature sayısı.
        nsamples: KernelExplainer kullanılıyorsa arka plan örnek sayısı.
    
    Returns:
        explainer: SHAP explainer nesnesi.
        shap_values: SHAP değerleri.
    """
    # Feature isimlerini ayarla
    if hasattr(X_train, 'columns'):
        feature_names = X_train.columns.tolist()
    elif feature_names is None:
        feature_names = [f"f{i}" for i in range(X_train.shape[1])]
    
    # Explainer seçimi
    try:
        explainer = shap.TreeExplainer(model)  # Ağacı destekliyorsa
    except Exception:
        background = shap.sample(X_train, nsamples)
        explainer = shap.KernelExplainer(model.predict, background)
    
    # SHAP değerlerini hesapla
    shap_values = explainer.shap_values(X_test)
    
    # Summary bar plot
    plt.figure(figsize=(8, 6))
    shap.summary_plot(shap_values, X_test, feature_names=feature_names,
                      max_display=max_display, plot_type="bar", show=True)
    
    # Summary dot plot
    plt.figure(figsize=(8, 6))
    shap.summary_plot(shap_values, X_test, feature_names=feature_names,
                      max_display=max_display, show=True)
    
    return explainer, shap_values

# Örnek Kullanım:
# explainer, shap_values = shap_feature_analysis(trained_model, X_train, X_test)
# shap.dependence_plot("important_feature", shap_values, X_test, feature_names=X_test.columns)
# plt.show()
