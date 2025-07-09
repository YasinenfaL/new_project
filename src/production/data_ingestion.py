# LOAN MODEL CONFIG
loan:
  feature_selection:
    selected_features:
      - age
      - gender
      - education_level
      - marital_status
      - job_type
      - CityId
      - income
      - IsActiveProject
      - IsActive

  data_types:
    to_object:
      - gender
      - education_level
      - marital_status
      - job_type

  missing_values:
    fill_zero:
      - income

  region_mapping:
    plaka_column: CityId
    output_column: region
    Marmara: [34, 41, 59]
    Ege: [35, 48]
    Akdeniz: [1, 7, 31]
    İç_Anadolu: [6, 42]
    Karadeniz: [61, 52]
    Doğu_Anadolu: [25, 44]
    Güneydoğu_Anadolu: [27, 63]

  default_risk_scores:
    Marmara: 0.2
    Ege: 0.3
    Akdeniz: 0.4
    İç_Anadolu: 0.5
    Karadeniz: 0.45
    Doğu_Anadolu: 0.6
    Güneydoğu_Anadolu: 0.65
    Unknown: 1.0

  feature_mapping:
    column_mapping:
      education_level: education_mapping
      marital_status: marital_mapping
      job_type: job_mapping
      gender: gender_mapping
      IsActiveProject: isactiveproject_mapping
      IsActive: isactive_mapping
    education_mapping:
      "1": "Primary"
      "2": "Secondary"
      "3": "Higher"
    job_mapping:
      "1": "Worker"
      "2": "Officer"
      "3": "Manager"
    marital_mapping:
      "1": "Single"
      "2": "Married"
      "3": "Divorced"
    gender_mapping:
      "0": "Erkek"
      "1": "Kadın"
    isactiveproject_mapping:
      "0": "Hayır"
      "1": "Evet"
    isactive_mapping:
      "0": "Hayır"
      "1": "Evet"

  feature_encoding:
    categorical_cols:
      - gender
      - education_level
      - marital_status
      - job_type
      - region
      - IsActiveProject

  scaling:
    exclude_columns:
      - city_code
      - age

  model_input_selection:
    include_columns:
      - age
      - gender
      - education_level
      - marital_status
      - job_type
      - region
      - income
      - IsActiveProject
      - IsActive
      - risk_score

# CAR MODEL CONFIG
car:
  feature_selection:
    selected_features:
      - age
      - gender
      - education_level
      - marital_status
      - job_type
      - CityId
      - income
      - IsActiveProject

  data_types:
    to_object:
      - gender
      - education_level
      - marital_status
      - job_type

  missing_values:
    fill_zero:
      - income

  region_mapping:
    plaka_column: CityId
    output_column: region
    Marmara: [34, 41, 59]
    Ege: [35, 48]
    Akdeniz: [1, 7, 31]
    İç_Anadolu: [6, 42]
    Karadeniz: [61, 52]
    Doğu_Anadolu: [25, 44]
    Güneydoğu_Anadolu: [27, 63]

  default_risk_scores:
    Marmara: 0.15
    Ege: 0.25
    Akdeniz: 0.35
    İç_Anadolu: 0.45
    Karadeniz: 0.4
    Doğu_Anadolu: 0.55
    Güneydoğu_Anadolu: 0.6
    Unknown: 1.0

  feature_mapping:
    column_mapping:
      education_level: education_mapping
      marital_status: marital_mapping
      job_type: job_mapping
      gender: gender_mapping
      IsActiveProject: isactiveproject_mapping
    education_mapping:
      "1": "Primary"
      "2": "Secondary"
      "3": "Higher"
    job_mapping:
      "1": "Worker"
      "2": "Officer"
      "3": "Manager"
    marital_mapping:
      "1": "Single"
      "2": "Married"
      "3": "Divorced"
    gender_mapping:
      "0": "Erkek"
      "1": "Kadın"
    isactiveproject_mapping:
      "0": "Hayır"
      "1": "Evet"

  feature_encoding:
    categorical_cols:
      - gender
      - education_level
      - marital_status
      - job_type
      - region
      - IsActiveProject

  scaling:
    exclude_columns:
      - city_code
      - age

  model_input_selection:
    include_columns:
      - age
      - gender
      - education_level
      - marital_status
      - job_type
      - region
      - income
      - IsActiveProject
      - risk_score

# HOME MODEL CONFIG
home:
  feature_selection:
    selected_features:
      - age
      - gender
      - education_level
      - marital_status
      - job_type
      - CityId
      - income
      - IsActive

  data_types:
    to_object:
      - gender
      - education_level
      - marital_status
      - job_type

  missing_values:
    fill_zero:
      - income

  region_mapping:
    plaka_column: CityId
    output_column: region
    Marmara: [34, 41, 59]
    Ege: [35, 48]
    Akdeniz: [1, 7, 31]
    İç_Anadolu: [6, 42]
    Karadeniz: [61, 52]
    Doğu_Anadolu: [25, 44]
    Güneydoğu_Anadolu: [27, 63]

  default_risk_scores:
    Marmara: 0.1
    Ege: 0.2
    Akdeniz: 0.3
    İç_Anadolu: 0.4
    Karadeniz: 0.35
    Doğu_Anadolu: 0.5
    Güneydoğu_Anadolu: 0.55
    Unknown: 1.0

  feature_mapping:
    column_mapping:
      education_level: education_mapping
      marital_status: marital_mapping
      job_type: job_mapping
      gender: gender_mapping
      IsActive: isactive_mapping
    education_mapping:
      "1": "Primary"
      "2": "Secondary"
      "3": "Higher"
    job_mapping:
      "1": "Worker"
      "2": "Officer"
      "3": "Manager"
    marital_mapping:
      "1": "Single"
      "2": "Married"
      "3": "Divorced"
    gender_mapping:
      "0": "Erkek"
      "1": "Kadın"
    isactive_mapping:
      "0": "Hayır"
      "1": "Evet"

  feature_encoding:
    categorical_cols:
      - gender
      - education_level
      - marital_status
      - job_type
      - region

  scaling:
    exclude_columns:
      - city_code
      - age

  model_input_selection:
    include_columns:
      - age
      - gender
      - education_level
      - marital_status
      - job_type
      - region
      - income
      - IsActive
      - risk_score
