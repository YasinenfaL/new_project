import time
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import Ridge, Lasso, ElasticNet, HuberRegressor, LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (
    RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor,
    ExtraTreesRegressor, HistGradientBoostingRegressor, BaggingRegressor
)
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

def evaluate_regression_models(
    X, y,
    test_size=0.2,
    random_state=42,
    log_transform=True
):
    """
    X, y            : feature matrix and target vector
    test_size       : fraction of data to hold out for testing
    random_state    : random seed
    log_transform   : if True, apply log1p to y before training
                      and inverse-transform predictions for metrics
    """
    # train/test split
    X_train, X_test, y_train_raw, y_test_raw = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    # apply log transform if requested
    if log_transform:
        y_train = np.log1p(y_train_raw)
        y_test  = np.log1p(y_test_raw)
    else:
        y_train = y_train_raw
        y_test  = y_test_raw

    models = {
        "LinearRegression": LinearRegression(n_jobs=-1),
        "Ridge": Ridge(alpha=1.0, solver="auto", random_state=random_state),
        "Lasso": Lasso(alpha=0.01, max_iter=2000, random_state=random_state),
        "ElasticNet": ElasticNet(alpha=0.01, l1_ratio=0.5, max_iter=2000, random_state=random_state),
        "HuberRegressor": HuberRegressor(epsilon=1.35, max_iter=2000),
        "DecisionTree": DecisionTreeRegressor(max_depth=10, min_samples_leaf=5, random_state=random_state),
        "RandomForest": RandomForestRegressor(
            n_estimators=300, max_depth=10, min_samples_leaf=3,
            n_jobs=-1, random_state=random_state
        ),
        "ExtraTrees": ExtraTreesRegressor(
            n_estimators=300, max_depth=10, min_samples_leaf=3,
            n_jobs=-1, random_state=random_state
        ),
        "GradientBoosting": GradientBoostingRegressor(
            n_estimators=300, learning_rate=0.05,
            max_depth=5, subsample=0.9, random_state=random_state
        ),
        "HistGradientBoosting": HistGradientBoostingRegressor(
            max_iter=300, learning_rate=0.05,
            max_depth=6, random_state=random_state
        ),
        "AdaBoost": AdaBoostRegressor(
            n_estimators=300, learning_rate=0.05, random_state=random_state
        ),
        "Bagging": BaggingRegressor(
            n_estimators=100, random_state=random_state, n_jobs=-1
        ),
        "SVR": SVR(C=1.0, epsilon=0.1, kernel='rbf'),
        "KNeighbors": KNeighborsRegressor(
            n_neighbors=7, weights='distance', n_jobs=-1
        ),
        "XGBoost": XGBRegressor(
            n_estimators=300, learning_rate=0.05, max_depth=6,
            subsample=0.9, colsample_bytree=0.8,
            random_state=random_state, n_jobs=-1
        ),
        "LightGBM": LGBMRegressor(
            n_estimators=300, learning_rate=0.05, num_leaves=64,
            subsample=0.9, colsample_bytree=0.8,
            random_state=random_state, n_jobs=-1
        ),
        "CatBoost": CatBoostRegressor(
            iterations=300, depth=6, learning_rate=0.05,
            l2_leaf_reg=3, verbose=False, random_state=random_state
        )
    }

    for name, model in models.items():
        print(f"\n=== {name} ===")
        try:
            start = time.time()
            model.fit(X_train, y_train)
            elapsed = time.time() - start

            # raw predictions (possibly log-scale)
            y_train_pred = model.predict(X_train)
            y_test_pred  = model.predict(X_test)

            # inverse-transform back to original scale if needed
            if log_transform:
                y_train_pred_orig = np.expm1(y_train_pred)
                y_test_pred_orig  = np.expm1(y_test_pred)
                y_train_true_orig = y_train_raw
                y_test_true_orig  = y_test_raw
            else:
                y_train_pred_orig = y_train_pred
                y_test_pred_orig  = y_test_pred
                y_train_true_orig = y_train_raw
                y_test_true_orig  = y_test_raw

            # compute metrics on original scale
            r2_train = r2_score(y_train_true_orig, y_train_pred_orig)
            r2_test  = r2_score(y_test_true_orig, y_test_pred_orig)

            rmse_train = np.sqrt(mean_squared_error(y_train_true_orig, y_train_pred_orig))
            rmse_test  = np.sqrt(mean_squared_error(y_test_true_orig, y_test_pred_orig))

            mae_train = mean_absolute_error(y_train_true_orig, y_train_pred_orig)
            mae_test  = mean_absolute_error(y_test_true_orig, y_test_pred_orig)

            print(f"[Train] R²: {r2_train:.4f} | RMSE: {rmse_train:.4f} | MAE: {mae_train:.4f}")
            print(f"[Test ] R²: {r2_test:.4f} | RMSE: {rmse_test:.4f} | MAE: {mae_test:.4f}")
            print(f"Runtime : {elapsed:.2f} sec")

        except Exception as e:
            print(f"[!] {name} hata verdi: {e}")

# Örnek kullanım:
# X, y = ...  # özellikler ve hedef (ör. fiyat verisi)
# evaluate_regression_models(X, y, log_transform=True)
