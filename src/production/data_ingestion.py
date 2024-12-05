import os
import sys
import pandas as pd
import sqlite3  # SQL veritabanına bağlanmak için
from dataclasses import dataclass
from src.exception import CustomException
from src.logger import logging


@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join("artifacts", "train.csv")
    test_data_path: str = os.path.join("artifacts", "test.csv")
    raw_data_path: str = os.path.join("artifacts", "data.csv")
    database_name: str = "example.db"  # SQL veritabanı adı
    query: str = "SELECT * FROM students;"  # Veritabanından alınacak sorgu


class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion method or component")
        try:
            # SQL veritabanına bağlan
            logging.info(f"Connecting to database: {self.ingestion_config.database_name}")
            connection = sqlite3.connect(self.ingestion_config.database_name)

            # SQL sorgusunu çalıştır ve veriyi pandas DataFrame'e yükle
            logging.info("Executing SQL query")
            df = pd.read_sql_query(self.ingestion_config.query, connection)
            logging.info("Data fetched from SQL database")

            # Veriyi artifacts dizinine kaydetmek için gerekli klasörleri oluştur
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)

            # SQL'den alınan ham veriyi kaydet
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)
            logging.info(f"Raw data saved at {self.ingestion_config.raw_data_path}")

            # Veriyi train-test setlerine ayır
            logging.info("Train-test split initiated")
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)

            # Train ve test setlerini artifacts klasörüne kaydet
            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            logging.info(f"Train data saved at {self.ingestion_config.train_data_path}")

            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)
            logging.info(f"Test data saved at {self.ingestion_config.test_data_path}")

            # Bağlantıyı kapat
            connection.close()
            logging.info("SQL database connection closed")

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path,
            )
        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    # Data ingestion işlemini başlat
    obj = DataIngestion()
    train_data, test_data = obj.initiate_data_ingestion()

    # Daha sonraki adımları çağırabilirsiniz
    # Örnek:
    # data_transformation = DataTransformation()
    # train_arr, test_arr, _ = data_transformation.initiate_data_transformation(train_data, test_data)
    # modeltrainer = ModelTrainer()
    # print(modeltrainer.initiate_model_trainer(train_arr, test_arr))
