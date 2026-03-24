from src.neuronTracer.config.configuration import ConfigurationManager
from src.neuronTracer.components.model_diagnosis import modelDiagnostic
from src.neuronTracer import logger


STAGE_NAME = "Model Diagnosis"

class ModelDiagnosisPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        model_diagnosis_config = config.get_model_diagnosis_config()
        diagnosis = modelDiagnostic(config=model_diagnosis_config)
        diagnosis.verifyPair()
        device=diagnosis.sysConfig()
        diagnosis.dataInspect()
        return device




if __name__ == '__main__':
    try:
        logger.info(f"*******************")
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = ModelDiagnosisPipeline()
        device = obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e

