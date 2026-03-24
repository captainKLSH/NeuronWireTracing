from src.neuronTracer.config.configuration import ConfigurationManager
from src.neuronTracer.components.model_train import ModelTraining
from src.neuronTracer import logger


STAGE_NAME = "Model Training"

class ModelTrainingPipeline:
    def __init__(self, device, encoder,train_loader,val_loader):
        self.device = device
        self.encoder=encoder
        self.train_loader=train_loader
        self.val_loader=val_loader

    def main(self):
        config = ConfigurationManager()
        model_train_config = config.get_model_train_config()
        train = ModelTraining(config=model_train_config, encoder=self.encoder, device=self.device, train_loader=self.train_loader, val_loader=self.val_loader)
        train.training()




if __name__ == '__main__':
    try:
        logger.info(f"*******************")
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = ModelTrainingPipeline()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e

