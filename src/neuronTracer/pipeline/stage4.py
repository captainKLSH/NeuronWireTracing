from src.neuronTracer.config.configuration import ConfigurationManager
from src.neuronTracer.components.model_build import *
from src.neuronTracer import logger
from src.neuronTracer.components.model_build import DinoEncoder


STAGE_NAME = "Model Encoder Build"

class ModelEncoderPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        model_enc_config = config.get_model_build_enc_config()
        encoder = DinoEncoder()
        # logger.info(encoder.state_dict())
        train_loader, val_loader = build_dataloaders(model_enc_config)

        return encoder, train_loader,val_loader




if __name__ == '__main__':
    try:
        logger.info(f"*******************")
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = ModelEncoderPipeline()
        encoder, train_loader,val_loader=obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e

