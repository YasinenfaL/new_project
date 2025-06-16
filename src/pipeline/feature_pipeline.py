def run_pipeline(input_json):
    try:
        start_time = time.time()
        logger.info("Converting JSON data to DataFrame...")
        df = pd.DataFrame.from_dict(input_json)
        logger.info(f"JSON to DataFrame conversion took: {time.time() - start_time:.2f} seconds")

        processor = DataProcessor(config_path=config_path, category_mappings_dir=category_mappings_dir)

        step_start = time.time()
        logger.info("Selecting features...")
        df = processor.select_features(df)
        logger.info(f"Feature selection took: {time.time() - step_start:.2f} seconds")

        step_start = time.time()
        logger.info("Converting data types...")
        df = processor.convert_dtypes(df)
        logger.info(f"Data type conversion took: {time.time() - step_start:.2f} seconds")

        step_start = time.time()
        logger.info("Filling missing values...")
        df = processor.fill_missing_values(df)
        logger.info(f"Missing value filling took: {time.time() - step_start:.2f} seconds")

        step_start = time.time()
        logger.info("Processing feature mappings...")
        df = processor.process_feature_mapping(df)
        logger.info(f"Feature mapping processing took: {time.time() - step_start:.2f} seconds")

        step_start = time.time()
        logger.info("Processing region features...")
        df = processor.process_region_feature(df)
        logger.info(f"Region feature processing took: {time.time() - step_start:.2f} seconds")

        step_start = time.time()
        logger.info("Calculating default risk scores...")
        df = processor.calculate_default_risk_score(df)
        logger.info(f"Default risk score calculation took: {time.time() - step_start:.2f} seconds")

        step_start = time.time()
        logger.info("Encoding categorical features...")
        df = processor.encode_categorical_features(df)
        logger.info(f"Categorical feature encoding took: {time.time() - step_start:.2f} seconds")

        step_start = time.time()
        logger.info("Scaling numeric features...")
        df = processor.scale_numeric_features(df)
        logger.info(f"Numeric feature scaling took: {time.time() - step_start:.2f} seconds")

        step_start = time.time()
        logger.info("Selecting model input features...")
        processed_data = processor.model_input_select(df)
        logger.info(f"Model input selection took: {time.time() - step_start:.2f} seconds")

        step_start = time.time()
        logger.info("Loading model...")
        with open(model_path, "rb") as model_file:
            model = pickle.load(model_file)
        logger.info(f"Model loading took: {time.time() - step_start:.2f} seconds")

        step_start = time.time()
        logger.info("Making predictions...")
        predictions = model.predict(processed_data)
        logger.info(f"Prediction took: {time.time() - step_start:.2f} seconds")

        total_time = time.time() - start_time
        logger.info(f"Total pipeline execution time: {total_time:.2f} seconds")

        response = {
            "result": predictions[0],
            "errorMessage": None,
            "success": True
        }

        logger.info("Pipeline successfully executed.")
        return response

    except Exception as e:
        logger.error(f"Error occurred while running pipeline: {str(e)}")
        return {
            "result": None,
            "errorMessage": str(e),
            "success": False
        }
