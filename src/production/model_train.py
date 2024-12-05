import os
import sys
import yaml
import numpy as np
from dataclasses import dataclass
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier, AdaBoostClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object


def load_config(config_path: str) -> dict:
    """YAML konfigürasyon dosyasını yükler."""
    try:
        with open(config_path, "r") as file:
            return yaml.safe_load(file)
    except Exception as e:
        raise CustomException(e, sys)


@dataclass
class ModelTrainerConfig:
    config_path: str = "config.yaml"

    # Konfigürasyon dosyasından değişkenler
    test_size: float = None
    random_state: int = None
    trained_model_file_path: str = None
    models: dict = None

    def __post_init__(self):
        """Konfigürasyon dosyasını okuyarak değişkenleri doldurur."""
        config = load_config(self.config_path)
        self.test_size = config["data_split"]["test_size"]
        self.random_state = config["data_split"]["random_state"]
        self.trained_model_file_path = config["output"]["trained_model_file"]
        self.models = config["models"]


class ModelTrainer:
    def __init__(self):
        self.config = ModelTrainerConfig()

    def initiate_model_trainer(self, X, y):
        try:
            logging.info("Splitting data into train and test sets")
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=self.config.test_size, random_state=self.config.random_state
            )

            models = {
                "Gradient Boosting": GradientBoostingClassifier(),
                "XGBoost Classifier": XGBClassifier(use_label_encoder=False, eval_metric='logloss'),
                "CatBoost Classifier": CatBoostClassifier(verbose=False),
                "AdaBoost Classifier": AdaBoostClassifier(),
            }

            logging.info("Starting model evaluation")
            best_model = None
            best_model_name = None
            best_test_accuracy = 0

            for model_name, model in models.items():
                logging.info(f"Training {model_name}")

                # Train the model
                model.fit(X_train, y_train)
                train_preds = model.predict(X_train)
                test_preds = model.predict(X_test)

                # Compute metrics
                train_accuracy = accuracy_score(y_train, train_preds)
                test_accuracy = accuracy_score(y_test, test_preds)
                train_precision = precision_score(y_train, train_preds, average='weighted')
                test_precision = precision_score(y_test, test_preds, average='weighted')
                train_recall = recall_score(y_train, train_preds, average='weighted')
                test_recall = recall_score(y_test, test_preds, average='weighted')
                train_f1 = f1_score(y_train, train_preds, average='weighted')
                test_f1 = f1_score(y_test, test_preds, average='weighted')

                # Log the metrics
                logging.info(f"{model_name} -> Train Metrics: Accuracy: {train_accuracy:.4f}, "
                             f"Precision: {train_precision:.4f}, Recall: {train_recall:.4f}, F1 Score: {train_f1:.4f}")
                logging.info(f"{model_name} -> Test Metrics: Accuracy: {test_accuracy:.4f}, "
                             f"Precision: {test_precision:.4f}, Recall: {test_recall:.4f}, F1 Score: {test_f1:.4f}")

                # Save the best model
                if test_accuracy > best_test_accuracy:
                    best_model = model
                    best_model_name = model_name
                    best_test_accuracy = test_accuracy

            if best_model is None:
                raise CustomException("No suitable model found")

            logging.info(f"Best model: {best_model_name} with Test Accuracy: {best_test_accuracy:.4f}")

            # Save the best model
            save_object(
                file_path=self.config.trained_model_file_path,
                obj=best_model
            )
            logging.info(f"Best model saved at {self.config.trained_model_file_path}")

            return best_model_name, best_test_accuracy

        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":

    trainer = ModelTrainer()
    best_model_name, best_test_accuracy = trainer.initiate_model_trainer(X, y)

    print(f"Best Model: {best_model_name}, Test Accuracy: {best_test_accuracy}")
