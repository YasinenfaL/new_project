from dataclasses import dataclass, field
from pathlib import Path
import yaml
from typing import List, Dict

@dataclass
class FeatureSelectionConfig:
    selected_features: List[str]

@dataclass
class DataTypeConfig:
    to_object: List[str]

@dataclass
class MissingValuesConfig:
    fill_zero: List[str]

@dataclass
class RegionMappingConfig:
    plaka_column: str
    output_column: str
    Marmara: List[int]
    Ege: List[int]
    Akdeniz: List[int]
    İç_Anadolu: List[int]
    Karadeniz: List[int]
    Doğu_Anadolu: List[int]
    Güneydoğu_Anadolu: List[int]

    def get_region(self, plaka: int) -> str:
        regions = {
            'Marmara': self.Marmara,
            'Ege': self.Ege,
            'Akdeniz': self.Akdeniz,
            'İç Anadolu': self.İç_Anadolu,
            'Karadeniz': self.Karadeniz,
            'Doğu Anadolu': self.Doğu_Anadolu,
            'Güneydoğu Anadolu': self.Güneydoğu_Anadolu
        }
        for region, codes in regions.items():
            if plaka in codes:
                return region
        return 'Bilinmeyen'

@dataclass
class DefaultRiskScoresConfig:
    Marmara: float
    Ege: float
    Akdeniz: float
    İç_Anadolu: float
    Karadeniz: float
    Doğu_Anadolu: float
    Güneydoğu_Anadolu: float
    Unknown: float

    def get_risk_score(self, region: str) -> float:
        normalized_region = region.replace(' ', '_')
        return getattr(self, normalized_region, self.Unknown)

@dataclass
class FeatureMappingConfig:
    column_mapping: Dict[str, str]
    education_mapping: Dict[str, str]
    job_mapping: Dict[str, str]
    marital_mapping: Dict[str, str]
    gender_mapping: Dict[str, str]
    isactiveproject_mapping: Dict[str, str] = field(default_factory=dict)
    isactive_mapping: Dict[str, str] = field(default_factory=dict)

    def get_mapping_dict(self, mapping_name: str) -> Dict[str, str]:
        return getattr(self, mapping_name, {})

@dataclass
class FeatureEncodingConfig:
    categorical_cols: List[str]

@dataclass
class ScalingConfig:
    exclude_columns: List[str]

@dataclass
class ModelInputSelectionConfig:
    include_columns: List[str]

class ConfigManager:
    def __init__(self, config_path: Path = None, model_type: str = "loan"):
        self.config_path = config_path or (Path(__file__).parent / 'config.yaml')
        self.model_type = model_type
        self.load_config()

    def load_config(self):
        with open(self.config_path, 'r', encoding='utf-8') as file:
            cfg = yaml.safe_load(file)

        model_cfg = cfg[self.model_type]
        
        self.feature_selection = FeatureSelectionConfig(**model_cfg['feature_selection'])
        self.data_types = DataTypeConfig(**model_cfg['data_types'])
        self.missing_values = MissingValuesConfig(**model_cfg['missing_values'])
        self.region_mapping = RegionMappingConfig(**model_cfg['region_mapping'])
        self.default_risk_scores = DefaultRiskScoresConfig(**model_cfg['default_risk_scores'])
        self.feature_mapping = FeatureMappingConfig(**model_cfg['feature_mapping'])
        self.feature_encoding = FeatureEncodingConfig(**model_cfg['feature_encoding'])
        self.scaling = ScalingConfig(**model_cfg['scaling'])
        self.model_input_selection = ModelInputSelectionConfig(**model_cfg['model_input_selection'])

    def reload(self):
        self.load_config()
