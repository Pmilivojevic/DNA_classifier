from dnaseq.config.configuration import ConfigurationManager
from dnaseq.components.model_evaluation import ModelEvaluation
from dnaseq import logger


STAGE_NAME = "Model Evaluation"

class ModelEvluationTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        model_evaluation_config = config.get_model_evaluation_config()
        model_evaluation = ModelEvaluation(config=model_evaluation_config)
        model_evaluation.evaluate()


if __name__ == "__main__":
    try:
        logger.info(f">>>>>>>>>>>>>>>>> stage {STAGE_NAME} started <<<<<<<<<<<<<<<<<")
        obj = ModelEvluationTrainingPipeline()
        obj.main()
        logger.info(f">>>>>>>>>>>>>>>>> stage {STAGE_NAME} completed <<<<<<<<<<<<<<<<<")

    except Exception as e:
        logger.exception(e)
        raise e
