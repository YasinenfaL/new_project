import os
import sys
import pandas as pd
import pickle
import logging
import time

# Proje kök dizinini Python path'ine ekle
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)

from src.process.data_processor import DataProcessor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

config_path = os.path.join(project_root, "config.yaml")
category_mappings_dir = os.path.join(project_root, "category_mappings")
loan_model_path = os.path.join(project_root, "loan_model.pkl")
car_model_path = os.path.join(project_root, "car_model.pkl")
home_model_path = os.path.join(project_root, "home_model.pkl")


def process_model_data(input_json, model_type):
    """Her model için veri işleme"""
    df = pd.DataFrame.from_dict(input_json)
    model_category_mappings_dir = os.path.join(category_mappings_dir, model_type)
    processor = DataProcessor(config_path=config_path, category_mappings_dir=model_category_mappings_dir, model_type=model_type)
    
    logger.info(f"Processing data for {model_type} model...")
    df = processor.select_features(df)
    df = processor.convert_dtypes(df)
    df = processor.fill_missing_values(df)
    df = processor.process_feature_mapping(df)
    df = processor.process_region_feature(df)
    df = processor.calculate_default_risk_score(df)
    df = processor.encode_categorical_features(df)
    df = processor.log_transform_features(df)
    processed_data = processor.model_input_select(df)
    
    return processed_data, processor


def run_pipeline(input_json):
    try:
        start_time = time.time()
        logger.info("Starting multi-model pipeline...")
        
        results = {}
        
        # LOAN MODEL
        logger.info("Processing loan model...")
        loan_start = time.time()
        loan_data, loan_processor = process_model_data(input_json, "loan")
        with open(loan_model_path, "rb") as model_file:
            loan_model = pickle.load(model_file)
        loan_predictions = loan_model.predict(loan_data)
        loan_predictions_df = pd.DataFrame({'predictions': loan_predictions})
        loan_transformed = loan_processor.inverse_log_transform(loan_predictions_df)
        results['loan_predict'] = loan_transformed['predictions'].values[0]
        loan_time = time.time() - loan_start
        logger.info(f"Loan model total time: {loan_time:.2f} seconds")
        
        # CAR MODEL
        logger.info("Processing car model...")
        car_start = time.time()
        car_data, car_processor = process_model_data(input_json, "car")
        with open(car_model_path, "rb") as model_file:
            car_model = pickle.load(model_file)
        car_predictions = car_model.predict(car_data)
        car_predictions_df = pd.DataFrame({'predictions': car_predictions})
        car_transformed = car_processor.inverse_log_transform(car_predictions_df)
        results['car_predict'] = car_transformed['predictions'].values[0]
        car_time = time.time() - car_start
        logger.info(f"Car model total time: {car_time:.2f} seconds")
        
        # HOME MODEL
        logger.info("Processing home model...")
        home_start = time.time()
        home_data, home_processor = process_model_data(input_json, "home")
        with open(home_model_path, "rb") as model_file:
            home_model = pickle.load(model_file)
        home_predictions = home_model.predict(home_data)
        home_predictions_df = pd.DataFrame({'predictions': home_predictions})
        home_transformed = home_processor.inverse_log_transform(home_predictions_df)
        results['home_predict'] = home_transformed['predictions'].values[0]
        home_time = time.time() - home_start
        logger.info(f"Home model total time: {home_time:.2f} seconds")

        total_time = time.time() - start_time
        logger.info(f"Total pipeline execution time: {total_time:.2f} seconds")

        response = {
            "loan_predict": results['loan_predict'],
            "car_predict": results['car_predict'],
            "home_predict": results['home_predict'],
            "timing_info": {
                "total_time": round(total_time, 2),
                "loan_model_time": round(loan_time, 2),
                "car_model_time": round(car_time, 2),
                "home_model_time": round(home_time, 2)
            },
            "errorMessage": None,
            "success": True
        }

        logger.info("Multi-model pipeline successfully executed.")
        return response

    except Exception as e:
        logger.error(f"Error occurred while running pipeline: {str(e)}")
        return {
            "loan_predict": None,
            "car_predict": None,
            "home_predict": None,
            "timing_info": None,
            "errorMessage": str(e),
            "success": False
        }
