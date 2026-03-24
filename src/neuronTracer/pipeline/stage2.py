from src.neuronTracer.config.configuration import ConfigurationManager
from src.neuronTracer.components.data_transform import DataTransform
from src.neuronTracer import logger


STAGE_NAME = "Data Transformation"

class DataTransformPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        data_transform_config = config.get_data_transform_config()
        data_transform = DataTransform(config=data_transform_config)
        data_transform.repackage_tiff()





if __name__ == '__main__':
    try:
        logger.info(f"*******************")
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = DataTransformPipeline()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e

