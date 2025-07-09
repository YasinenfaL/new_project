import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pickle, gzip


# --- 2) Optuna objective fonksiyonu ---
def objective(trial):
    param = {
        'objective': 'regression',
        'metric': 'mae',
        'verbosity': -1,
        'boosting_type': 'gbdt',
        'n_jobs': -1,
        'random_state': 42,
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000, step=100),
        'learning_rate': trial.suggest_loguniform('learning_rate', 1e-3, 0.3),
        'num_leaves': trial.suggest_int('num_leaves', 16, 256),
        'max_depth': trial.suggest_int('max_depth', 3, 16),
        'subsample': trial.suggest_uniform('subsample', 0.5, 1.0),
        'subsample_freq': trial.suggest_int('subsample_freq', 1, 10),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
        'reg_alpha': trial.suggest_loguniform('reg_alpha', 1e-8, 10.0),
        'reg_lambda': trial.suggest_loguniform('reg_lambda', 1e-8, 10.0),
    }

    gbm = lgb.LGBMRegressor(**param)
    # Negatif MAE döner; maximize edeceğiz
    scores = cross_val_score(
        gbm, X_train, y_train_log,
        scoring='neg_mean_absolute_error', cv=5
    )
    return scores.mean()

# --- 3) Optuna Study ---
study = optuna.create_study(
    direction="maximize",
    sampler=TPESampler(seed=42),
    pruner=MedianPruner(n_startup_trials=5)
)
study.optimize(objective, n_trials=50, show_progress_bar=True)

print("► En iyi skor (neg MAE):", study.best_value)
print("► En iyi parametreler:", study.best_params)







# --- 1) Veri Hazırlığı ---
# df: DataFrame'iniz
# target_column: Hedef sütun adı
# Örn: df = pd.read_csv('data.csv'); target_column = 'sales'
X = df.drop(target_column, axis=1)
y_raw = df[target_column]

# Log1p dönüşümü (log ölçeğine alma)
y = np.log1p(y_raw)

# --- 2) Train/Test Ayrımı (son %20'si test) ---
X_train, X_test, y_train_log, y_test_log = train_test_split(
    X, y, test_size=0.2, shuffle=False
)

# --- 3) İlk Model: Feature Importance için Eğitim ---
best_params = {
    'learning_rate': 0.05,
    'num_leaves': 31,
    'n_estimators': 200,
    'max_depth': -1,
    'min_child_samples': 20,
    # ... diğer en iyi parametreleriniz
}
model_fs = lgb.LGBMRegressor(**best_params)
model_fs.fit(X_train, y_train_log)

# Normalize edilmiş feature importances
importances = pd.Series(model_fs.feature_importances_, index=X_train.columns)
importances /= importances.sum()

# 0.01'den büyük olan özellikleri seç
selected_features = importances[importances > 0.01].index.tolist()
print("Seçilen Özellikler:", selected_features)

# Alt kümeleri oluştur
X_train_sel = X_train[selected_features]
X_test_sel  = X_test[selected_features]

# --- 4) Nihai Model Eğitimi ---
model = lgb.LGBMRegressor(**best_params)
model.fit(X_train_sel, y_train_log)

# --- 5) Tahminler (log ölçekli) ---
y_train_pred_log = model.predict(X_train_sel)
y_test_pred_log  = model.predict(X_test_sel)

# --- 6) Orijinal Ölçeğe Geri Dönüşüm ---
inv_func = np.expm1  # log1p'in inverse fonksiyonu
y_train_true = inv_func(y_train_log)
y_train_pred = inv_func(y_train_pred_log)
y_test_true  = inv_func(y_test_log)
y_test_pred  = inv_func(y_test_pred_log)

# --- 7) Metrik Hesaplama ---
def compute_metrics(y_true, y_pred):
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    r2   = r2_score(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    return mae, rmse, r2, mape

train_mae, train_rmse, train_r2, train_mape = compute_metrics(y_train_true, y_train_pred)
test_mae,  test_rmse,  test_r2,  test_mape  = compute_metrics(y_test_true,  y_test_pred)

# Sonuçları yazdır
print("\n=== TRAIN METRİKLERİ ===")
print(f"MAE:  {train_mae:.4f}")
print(f"RMSE: {train_rmse:.4f}")
print(f"R²:   {train_r2:.4f}")
print(f"MAPE: {train_mape:.2f}%")

print("\n=== TEST METRİKLERİ ===")
print(f"MAE:  {test_mae:.4f}")
print(f"RMSE: {test_rmse:.4f}")
print(f"R²:   {test_r2:.4f}")
print(f"MAPE: {test_mape:.2f}%")

# --- 8) Modeli Gzip ile Kaydetme ---
with gzip.open('lgbm_model_selected.pkl.gz', 'wb') as f:
    pickle.dump(model, f)

print("\nModel 'lgbm_model_selected.pkl.gz' olarak kaydedildi.")
