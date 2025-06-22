import time
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
from imblearn.ensemble import (
    BalancedBaggingClassifier,
    BalancedRandomForestClassifier,
    EasyEnsembleClassifier,
    RUSBoostClassifier
)
from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    BaggingClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    AdaBoostClassifier
)
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier, PassiveAggressiveClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
import numpy as np

def evaluate_models(
    X, y, test_size=0.2, threshold=0.7, random_state=42
):
    """
    Dengesiz veri setlerinde çeşitli modelleri deneyerek:
      - Stratify edilmiş train/test split
      - Prob>threshold ise 1, değilse 0
      - Geniş hiperparametre setleri
    Her model için classification report, ROC-AUC, Gini, runtime basar.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size,
        stratify=y, random_state=random_state
    )

    models = {
        'BalancedBagging': BalancedBaggingClassifier(
            base_estimator=DecisionTreeClassifier(),
            n_estimators=50,
            sampling_strategy='auto',
            replacement=False,
            random_state=random_state
        ),
        'BalancedRandomForest': BalancedRandomForestClassifier(
            n_estimators=200,
            max_depth=None,
            n_jobs=-1,
            random_state=random_state
        ),
        'EasyEnsemble': EasyEnsembleClassifier(
            n_estimators=20,
            random_state=random_state
        ),
        'RUSBoost': RUSBoostClassifier(
            n_estimators=200,
            random_state=random_state
        ),
        'RandomForest': RandomForestClassifier(
            n_estimators=500,
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            max_features='sqrt',
            class_weight='balanced',
            n_jobs=-1,
            random_state=random_state
        ),
        'ExtraTrees': ExtraTreesClassifier(
            n_estimators=500,
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            max_features='sqrt',
            class_weight='balanced',
            n_jobs=-1,
            random_state=random_state
        ),
        'GradientBoosting': GradientBoostingClassifier(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=5,
            subsample=0.8,
            random_state=random_state
        ),
        'HistGradientBoosting': HistGradientBoostingClassifier(
            max_iter=500,
            learning_rate=0.05,
            max_depth=5,
            random_state=random_state
        ),
        'AdaBoost': AdaBoostClassifier(
            n_estimators=200,
            learning_rate=0.05,
            random_state=random_state
        ),
        'XGBoost': XGBClassifier(
            n_estimators=500,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=(y==0).sum()/(y==1).sum(),
            reg_alpha=0.1,
            reg_lambda=1,
            use_label_encoder=False,
            eval_metric='logloss',
            random_state=random_state,
            n_jobs=-1
        ),
        'LightGBM': LGBMClassifier(
            n_estimators=500,
            num_leaves=64,
            max_depth=-1,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            is_unbalance=True,
            random_state=random_state,
            n_jobs=-1
        ),
        'CatBoost': CatBoostClassifier(
            iterations=500,
            depth=6,
            learning_rate=0.05,
            l2_leaf_reg=3,
            bagging_temperature=1,
            auto_class_weights='Balanced',
            verbose=False,
            random_state=random_state
        ),
        'LogisticRegression': LogisticRegression(
            C=1.0,
            class_weight='balanced',
            max_iter=1000,
            n_jobs=-1,
            random_state=random_state
        ),
        'LinearSVC': LinearSVC(
            C=1.0,
            class_weight='balanced',
            max_iter=20000,
            random_state=random_state
        ),
        'SVC': SVC(
            C=1.0,
            probability=True,
            class_weight='balanced',
            random_state=random_state
        ),
        'RidgeClassifier': RidgeClassifier(
            alpha=1.0,
            class_weight='balanced',
            random_state=random_state
        ),
        'GaussianNB': GaussianNB(),
        'BernoulliNB': BernoulliNB(),
        'KNeighbors': KNeighborsClassifier(
            n_neighbors=7,
            weights='distance',
            n_jobs=-1
        ),
        'DecisionTree': DecisionTreeClassifier(
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            class_weight='balanced',
            random_state=random_state
        ),
        'QDA': QuadraticDiscriminantAnalysis(),
        'PassiveAggressive': PassiveAggressiveClassifier(
            max_iter=1000,
            random_state=random_state
        )
    }

    for name, model in models.items():
        print(f"\n=== {name} ===")
        try:
            start = time.time()
            model.fit(X_train, y_train)
            elapsed = time.time() - start

            # Olasılık tahminine göre sınıflandırma
            if hasattr(model, "predict_proba"):
                y_proba = model.predict_proba(X_test)[:, 1]
                y_pred = np.where(y_proba > threshold, 1, 0)
            elif hasattr(model, "decision_function"):
                y_scores = model.decision_function(X_test)
                y_pred = np.where(y_scores > threshold, 1, 0)
                y_proba = y_scores
            else:
                y_pred = model.predict(X_test)
                y_proba = None

            print("Classification Report:")
            print(classification_report(y_test, y_pred))

            if y_proba is not None:
                auc = roc_auc_score(y_test, y_proba)
                gini = 2 * auc - 1
                print(f"ROC AUC: {auc:.4f}")
                print(f"Gini: {gini:.4f}")

            print(f"Runtime: {elapsed:.2f} sec")

        except Exception as e:
            print(f"[!] {name} hata verdi: {e}")


# Örneğin CSV’den yükleme:
df = pd.read_csv("veri.csv")
X = df.drop("hedef", axis=1)
y = df["hedef"]

evaluate_models(X, y, test_size=0.25, threshold=0.7)
