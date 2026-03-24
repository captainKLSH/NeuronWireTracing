from src.neuronTracer.config.configuration import ConfigurationManager
from src.neuronTracer.components.data_convert import TorchConvert
from src.neuronTracer import logger


STAGE_NAME = "Data Torch Convert"

class DataConvertPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        torch_config = config.get_torch_config()
        torch_convert = TorchConvert(config=torch_config)
        torch_convert.extract_and_store_tensors()





if __name__ == '__main__':
    try:
        logger.info(f"*******************")
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = DataConvertPipeline()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e

