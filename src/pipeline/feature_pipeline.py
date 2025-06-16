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
model_path = os.path.join(project_root, "model_path.pkl")


def run_pipeline(input_json):
    try:
        timing_info = {}
        start_time = time.time()
        logger.info("Converting JSON data to DataFrame...")
        df = pd.DataFrame.from_dict(input_json)
        timing_info['json_to_df'] = time.time() - start_time
        logger.info(f"JSON to DataFrame conversion took: {timing_info['json_to_df']:.2f} seconds")

        processor = DataProcessor(config_path=config_path, category_mappings_dir=category_mappings_dir)

        step_start = time.time()
        logger.info("Selecting features...")
        df = processor.select_features(df)
        timing_info['feature_selection'] = time.time() - step_start
        logger.info(f"Feature selection took: {timing_info['feature_selection']:.2f} seconds")

        step_start = time.time()
        logger.info("Converting data types...")
        df = processor.convert_dtypes(df)
        timing_info['dtype_conversion'] = time.time() - step_start
        logger.info(f"Data type conversion took: {timing_info['dtype_conversion']:.2f} seconds")

        step_start = time.time()
        logger.info("Filling missing values...")
        df = processor.fill_missing_values(df)
        timing_info['missing_values'] = time.time() - step_start
        logger.info(f"Missing value filling took: {timing_info['missing_values']:.2f} seconds")

        step_start = time.time()
        logger.info("Processing feature mappings...")
        df = processor.process_feature_mapping(df)
        timing_info['feature_mapping'] = time.time() - step_start
        logger.info(f"Feature mapping processing took: {timing_info['feature_mapping']:.2f} seconds")

        step_start = time.time()
        logger.info("Processing region features...")
        df = processor.process_region_feature(df)
        timing_info['region_features'] = time.time() - step_start
        logger.info(f"Region feature processing took: {timing_info['region_features']:.2f} seconds")

        step_start = time.time()
        logger.info("Calculating default risk scores...")
        df = processor.calculate_default_risk_score(df)
        timing_info['risk_score'] = time.time() - step_start
        logger.info(f"Default risk score calculation took: {timing_info['risk_score']:.2f} seconds")

        step_start = time.time()
        logger.info("Encoding categorical features...")
        df = processor.encode_categorical_features(df)
        timing_info['categorical_encoding'] = time.time() - step_start
        logger.info(f"Categorical feature encoding took: {timing_info['categorical_encoding']:.2f} seconds")

        step_start = time.time()
        logger.info("Scaling numeric features...")
        df = processor.scale_numeric_features(df)
        timing_info['numeric_scaling'] = time.time() - step_start
        logger.info(f"Numeric feature scaling took: {timing_info['numeric_scaling']:.2f} seconds")

        step_start = time.time()
        logger.info("Selecting model input features...")
        processed_data = processor.model_input_select(df)
        timing_info['model_input_selection'] = time.time() - step_start
        logger.info(f"Model input selection took: {timing_info['model_input_selection']:.2f} seconds")

        step_start = time.time()
        logger.info("Loading model...")
        with open(model_path, "rb") as model_file:
            model = pickle.load(model_file)
        timing_info['model_loading'] = time.time() - step_start
        logger.info(f"Model loading took: {timing_info['model_loading']:.2f} seconds")

        step_start = time.time()
        logger.info("Making predictions...")
        predictions = model.predict(processed_data)
        timing_info['prediction'] = time.time() - step_start
        logger.info(f"Prediction took: {timing_info['prediction']:.2f} seconds")

        total_time = time.time() - start_time
        timing_info['total_time'] = total_time
        logger.info(f"Total pipeline execution time: {total_time:.2f} seconds")

        response = {
            "result": predictions[0],
            "timing_info": {k: round(v, 2) for k, v in timing_info.items()},
            "errorMessage": None,
            "success": True
        }

        logger.info("Pipeline successfully executed.")
        return response

    except Exception as e:
        logger.error(f"Error occurred while running pipeline: {str(e)}")
        return {
            "result": None,
            "timing_info": None,
            "errorMessage": str(e),
            "success": False
        }
