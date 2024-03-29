from dnaseq import logger
from dnaseq.pipelines.stage_01_data_ingestion import DataIngestionTrainingPipeline
from dnaseq.pipelines.stage_02_data_validation import DataValidationTrainingPipeline
from dnaseq.pipelines.stage_03_data_transformation import DataTransformationTrainingPipeline


STAGE_NAME = "Data Ingestion"

try:
    logger.info(f">>>>>>>>>>>>>>>>> stage {STAGE_NAME} started <<<<<<<<<<<<<<<<<")
    data_ingestion = DataIngestionTrainingPipeline()
    data_ingestion.main()
    logger.info(f">>>>>>>>>>>>>>>>> stage {STAGE_NAME} completed <<<<<<<<<<<<<<<<<")
    
except Exception as e:
    logger.exception(e)
    raise e


STAGE_NAME = "Data Validation"

try:
    logger.info(f">>>>>>>>>>>>>>>>> stage {STAGE_NAME} started <<<<<<<<<<<<<<<<<")
    data_validation = DataValidationTrainingPipeline()
    data_validation.main()
    logger.info(f">>>>>>>>>>>>>>>>> stage {STAGE_NAME} completed <<<<<<<<<<<<<<<<<")
    
except Exception as e:
    logger.exception(e)
    raise e


STAGE_NAME = "Data Transformation"

try:
    logger.info(f">>>>>>>>>>>>>>>>> stage {STAGE_NAME} started <<<<<<<<<<<<<<<<<")
    data_transformation = DataTransformationTrainingPipeline()
    data_transformation.main()
    logger.info(f">>>>>>>>>>>>>>>>> stage {STAGE_NAME} completed <<<<<<<<<<<<<<<<<")
    
except Exception as e:
    logger.exception(e)
    raise e
