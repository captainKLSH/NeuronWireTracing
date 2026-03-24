from src.neuronTracer.config.configuration import ConfigurationManager
from src.neuronTracer.components.model_build import *
from src.neuronTracer import logger
from src.neuronTracer.components.test import Test


STAGE_NAME = "Model Test Build"

class ModelTestPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        test_config = config.get_test_config()
        test = Test(config=test_config)
        test.take_final()
        test.run_official_evaluation()




if __name__ == '__main__':
    try:
        logger.info(f"*******************")
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = ModelTestPipeline()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e

