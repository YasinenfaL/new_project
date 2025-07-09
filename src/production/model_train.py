import os
import sys
from typing import List, Optional, Dict
from pathlib import Path
import pandas as pd
import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler

# Proje kök dizinini Python path'ine ekle
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)

from src.configs.config import ConfigManager

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))


class DataProcessor:
    def __init__(self, config_path: Optional[Path] = None, category_mappings_dir: Optional[str] = None, model_type: str = "loan"):
        try:
            self.model_type = model_type
            self.config_manager = ConfigManager(config_path, model_type)
            self.scaler = StandardScaler()
            self.category_mappings_dir = Path(category_mappings_dir) if category_mappings_dir else Path(os.path.dirname(SCRIPT_DIR)) / "mapping"
            if not self.category_mappings_dir.exists():
                raise FileNotFoundError(f"Mapping directory not found: {self.category_mappings_dir}")
        except Exception as e:
            raise Exception(f"DataProcessor initialization error: {str(e)}")

    def select_features(self, dataframe: pd.DataFrame, features: Optional[List[str]] = None) -> pd.DataFrame:
        try:
            features = features or self.config_manager.feature_selection.selected_features
            missing_cols = [col for col in features if col not in dataframe.columns]
            if missing_cols:
                raise ValueError(f"Required columns not found in data: {missing_cols}")
            return dataframe[features].copy()
        except Exception as e:
            raise Exception(f"Feature selection error: {str(e)}")

    def convert_dtypes(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        try:
            to_object_cols = [col for col in self.config_manager.data_types.to_object if col in dataframe.columns]
            dataframe[to_object_cols] = dataframe[to_object_cols].astype('object')
            
            if 'application_hour' in dataframe.columns:
                dataframe['application_hour'] = pd.to_datetime(dataframe['application_hour'], format='%H:%M:%S').dt.hour
                
            return dataframe
        except Exception as e:
            raise Exception(f"Data type conversion error: {str(e)}")

    def fill_missing_values(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        try:
            fill_zero_cols = [col for col in self.config_manager.missing_values.fill_zero if col in dataframe.columns]
            dataframe[fill_zero_cols] = dataframe[fill_zero_cols].fillna(0)
            return dataframe
        except Exception as e:
            raise Exception(f"Missing value filling error: {str(e)}")

    def process_feature_mapping(self, dataframe: pd.DataFrame, columns: Optional[Dict[str, str]] = None) -> pd.DataFrame:
        try:
            dataframe_processed = dataframe.copy()
            columns = columns or self.config_manager.feature_mapping.column_mapping
            if not columns:
                raise ValueError("No column mapping information found")
            for column, mapping_name in columns.items():
                if column in dataframe_processed.columns:
                    mapping_dict = self.config_manager.feature_mapping.get_mapping_dict(mapping_name)
                    if not mapping_dict:
                        raise ValueError(f"No mapping information found for '{mapping_name}'")
                    dataframe_processed[column] = dataframe_processed[column].astype(str).map(mapping_dict).fillna(dataframe_processed[column])
                else:
                    raise ValueError(f"Column '{column}' not found in dataset")
            return dataframe_processed
        except Exception as e:
            raise Exception(f"Feature mapping error: {str(e)}")

    def process_region_feature(self, dataframe: pd.DataFrame, plaka_column: Optional[str] = None, output_column: Optional[str] = None) -> pd.DataFrame:
        try:
            plaka_column = plaka_column or self.config_manager.region_mapping.plaka_column
            output_column = output_column or self.config_manager.region_mapping.output_column
            if plaka_column not in dataframe.columns:
                raise ValueError(f"Plaka column '{plaka_column}' not found in dataset")
            dataframe_processed = dataframe.copy()
            dataframe_processed[plaka_column] = pd.to_numeric(dataframe_processed[plaka_column], errors='coerce')
            def get_region_safe(plaka):
                if pd.isna(plaka):
                    return "Unknown"
                try:
                    return self.config_manager.region_mapping.get_region(plaka)
                except ValueError:
                    return "Unknown"
            dataframe_processed[output_column] = dataframe_processed[plaka_column].apply(get_region_safe)
            return dataframe_processed
        except Exception as e:
            raise Exception(f"Region feature processing error: {str(e)}")

    def calculate_default_risk_score(self, dataframe: pd.DataFrame, region_column: Optional[str] = None, output_column: str = 'risk_score') -> pd.DataFrame:
        try:
            region_column = region_column or self.config_manager.region_mapping.output_column
            if region_column not in dataframe.columns:
                raise ValueError(f"Region column '{region_column}' not found in dataset")
            dataframe_processed = dataframe.copy()
            dataframe_processed[output_column] = dataframe_processed[region_column].apply(self.config_manager.default_risk_scores.get_risk_score)
            return dataframe_processed
        except Exception as e:
            raise Exception(f"Risk score calculation error: {str(e)}")
    
    def encode_categorical_features(self, dataframe: pd.DataFrame, categorical_cols: Optional[List[str]] = None) -> pd.DataFrame:
        try:
            categorical_cols = categorical_cols or self.config_manager.feature_encoding.categorical_cols
            dataframe_processed = dataframe.copy()
            for col in categorical_cols:
                if col in dataframe_processed.columns:
                    categories_path = self.category_mappings_dir / f"{col}_categories.pkl"
                    if not categories_path.exists():
                        raise FileNotFoundError(f"Category file not found for '{col}': {categories_path}")
                    categories = joblib.load(categories_path)
                    dataframe_processed[col] = pd.Categorical(dataframe_processed[col], categories=categories)
            dataframe_processed = pd.get_dummies(dataframe_processed, columns=categorical_cols, dummy_na=False)
            return dataframe_processed
        except Exception as e:
            raise Exception(f"Categorical feature encoding error: {str(e)}")

    def log_transform_features(self, dataframe: pd.DataFrame, exclude_columns: Optional[List[str]] = None) -> pd.DataFrame:
        try:
            exclude_columns = exclude_columns or self.config_manager.scaling.exclude_columns
            numeric_columns = dataframe.select_dtypes(include=['int64', 'float64']).columns.tolist()
            numeric_columns = [col for col in numeric_columns if col not in exclude_columns]
            if not numeric_columns:
                raise ValueError("No numeric columns found for log transformation")
            dataframe_copy = dataframe.copy()
            for col in numeric_columns:
                dataframe_copy[col] = np.log1p(dataframe_copy[col])
            return dataframe_copy
        except Exception as e:
            raise Exception(f"Log transformation error: {str(e)}")

    def inverse_log_transform(self, dataframe: pd.DataFrame, exclude_columns: Optional[List[str]] = None) -> pd.DataFrame:
        try:
            exclude_columns = exclude_columns or self.config_manager.scaling.exclude_columns
            numeric_columns = dataframe.select_dtypes(include=['int64', 'float64']).columns.tolist()
            numeric_columns = [col for col in numeric_columns if col not in exclude_columns]
            if not numeric_columns:
                raise ValueError("No numeric columns found for inverse log transformation")
            dataframe_copy = dataframe.copy()
            for col in numeric_columns:
                dataframe_copy[col] = np.expm1(dataframe_copy[col])
            return dataframe_copy
        except Exception as e:
            raise Exception(f"Inverse log transformation error: {str(e)}")
        
    def model_input_select(self, dataframe: pd.DataFrame, include_columns: Optional[List[str]] = None) -> pd.DataFrame:
        try:
            include_columns = include_columns or self.config_manager.model_input_selection.include_columns
            valid_columns = [col for col in include_columns if col in dataframe.columns]
            if not valid_columns:
                raise ValueError("No valid columns found for model input")
            if len(valid_columns) < len(include_columns):
                missing_cols = set(include_columns) - set(valid_columns)
                raise ValueError(f"Required columns missing for model input: {missing_cols}")
            return dataframe[valid_columns]
        except Exception as e:
            raise Exception(f"Model input selection error: {str(e)}")
