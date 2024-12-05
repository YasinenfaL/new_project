import sys
import os
import yaml
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object


def load_config(config_path: str) -> dict:
    """YAML konfigürasyon dosyasını yükler."""
    try:
        with open(config_path, "r") as file:
            config = yaml.safe_load(file)
        return config
    except Exception as e:
        raise CustomException(e, sys)


@dataclass
class DataTransformationConfig:
    config_path: str = "config.yaml"

    # YAML dosyasından yüklenen değerler
    numerical_columns: list = None
    categorical_columns: list = None
    target_column: str = None
    preprocessor_obj_file_path: str = None

    def __post_init__(self):
        """YAML dosyasını okuyarak değişkenleri doldur."""
        config = load_config(self.config_path)
        self.numerical_columns = config["data_transformation"]["numerical_columns"]
        self.categorical_columns = config["data_transformation"]["categorical_columns"]
        self.target_column = config["data_transformation"]["target_column"]
        self.preprocessor_obj_file_path = config["data_transformation"]["preprocessor_file_path"]


class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        """
        Veri dönüşüm nesnesini oluşturur.
        """
        try:
            num_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler())
                ]
            )

            cat_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("one_hot_encoder", OneHotEncoder()),
                    ("scaler", StandardScaler(with_mean=False))
                ]
            )

            logging.info(f"Categorical columns: {self.data_transformation_config.categorical_columns}")
            logging.info(f"Numerical columns: {self.data_transformation_config.numerical_columns}")

            preprocessor = ColumnTransformer(
                [
                    ("num_pipeline", num_pipeline, self.data_transformation_config.numerical_columns),
                    ("cat_pipelines", cat_pipeline, self.data_transformation_config.categorical_columns)
                ]
            )

            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path, test_path):
        """
        Eğitim ve test verilerine dönüşüm uygular.
        """
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Read train and test data completed")
            logging.info("Obtaining preprocessing object")

            preprocessing_obj = self.get_data_transformer_object()

            target_column_name = self.data_transformation_config.target_column

            input_feature_train_df = train_df.drop(columns=[target_column_name], axis=1)
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df.drop(columns=[target_column_name], axis=1)
            target_feature_test_df = test_df[target_column_name]

            logging.info("Applying preprocessing object on training and testing dataframes.")

            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            train_arr = np.c_[
                input_feature_train_arr, np.array(target_feature_train_df)
            ]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            logging.info("Saving preprocessing object.")

            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )
        except Exception as e:
            raise CustomException(e, sys)
