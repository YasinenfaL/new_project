import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from xgboost import XGBClassifier
import pickle
import gzip

# Configuration parameters
config = {
    'n_estimators': 100,
    'max_depth': 5,
    'learning_rate': 0.1,
    'min_child_weight': 3,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'gamma': 0.1,
    'reg_alpha': 0.1,
    'reg_lambda': 0.1,
    'use_label_encoder': False,
    'eval_metric': 'logloss',
    'random_state': 42
}

def train_model(features, target, config, test_size=0.20, threshold=0.7, random_state=42):
    # Split data
    x_train, x_test, y_train, y_test = train_test_split(
        features, target, test_size=test_size, random_state=random_state, stratify=target
    )

    # Train model
    model = XGBClassifier(**config)
    model.fit(x_train, y_train)

    # Make predictions
    y_pred_proba = model.predict_proba(x_test)[:, 1]
    y_pred = (y_pred_proba >= threshold).astype(int)

    # Print classification report
    report = classification_report(y_test, y_pred)
    print("Classification Report:\n", report)

    # Save predictions
    results_df = pd.DataFrame({
        'actual': y_test,
        'predicted': y_pred,
        'probability': y_pred_proba
    })
    results_df.to_csv('prediction_results.csv', index=False)

    # Save model with gzip compression
    with gzip.open('xgb_model.pkl.gz', 'wb') as f:
        pickle.dump(model, f)

    print("\nModel trained and saved as 'xgb_model.pkl.gz'")
    print("Results saved as 'prediction_results.csv'")

if __name__ == "__main__":
    # Load dataset
    data = pd.read_csv("data/processed/processed_data.csv")

    # Features and target
    target = data['target']
    features = data.drop('target', axis=1)

    # Train model
    train_model(features, target, config)
