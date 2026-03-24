from src.neuronTracer import *
from src.neuronTracer.constants import *
from src.neuronTracer.utils.common import read_yaml, create_directories
from src.neuronTracer.entity.config import (
    DataIngestionConfig,
    DataTransformConfig,
    ModelDiagnosisConfig,
    TorchConvertConfig,
    ModelBuildEncConfig,
    ModelTrainConfig,
    TestConfig
)


class ConfigurationManager:
    def __init__(
        self, config_filepath=CONFIG_FILE_PATH, params_filepath=PARAMS_FILE_PATH
    ):

        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)

        create_directories([self.config.artifacts_root])

    def get_data_ingestion_config(self) -> DataIngestionConfig:
        config = self.config.data_ingestion

        create_directories([config.root_dir])

        data_ingestion_config = DataIngestionConfig(
            root_dir=Path(config.root_dir),
            source_URL=Path(config.source_URL),
            local_data_file=Path(config.local_data_file),
            unzip_dir=Path(config.unzip_dir),
        )

        return data_ingestion_config

    def get_data_transform_config(self) -> DataTransformConfig:
        config = self.config.data_transform

        create_directories([config.root_dir])

        data_transform_config = DataTransformConfig(
            root_dir=Path(config.root_dir),
            train_data=Path(config.train_data),
            label_data=Path(config.label_data),
            test_data=Path(config.test_data),
            params_box_size=tuple(self.params.box_size_mac),
        )

        return data_transform_config

    def get_torch_config(self) -> TorchConvertConfig:
        config = self.config.torch_convert

        create_directories([config.root_dir])

        torch_config = TorchConvertConfig(
            root_dir=Path(config.root_dir),
            train_data=Path(config.train_data),
            label_data=Path(config.label_data),
            params_box_size=tuple(self.params.box_size_mac),
        )

        return torch_config

    def get_model_diagnosis_config(self) -> ModelDiagnosisConfig:
        config = self.config.model_diagnosis

        model_diagnosis_config = ModelDiagnosisConfig(
            train_img=Path(config.train_img),
            label_pth=Path(config.label_pth),
            params_box_size=tuple(self.params.box_size_mac),
        )
        return model_diagnosis_config

    def get_model_build_enc_config(self) -> ModelBuildEncConfig:
        config = self.config.model_build_enc

        model_build_enc_config = ModelBuildEncConfig(
            train_img=Path(config.train_img),
            label_pth=Path(config.label_pth),
            params_box_size=tuple(self.params.box_size_mac),
            params_batchSize=int(self.params.BATCH_SIZE),
            params_outputSize=int(self.params.OUTPUT_SIZE),
            params_epoch=int(self.params.EPOCHS),
            params_lr=float(self.params.LR),
            params_augmentT=bool(self.params.augmentTrain),
            params_augmentL=bool(self.params.augmentLabel),
        )
        return model_build_enc_config

    def get_model_train_config(self) -> ModelTrainConfig:
        config = self.config.model_train

        create_directories([config.root_dir])

        model_train_config = ModelTrainConfig(
            root_dir=Path(config.root_dir),
            train_img=Path(config.train_img),
            label_pth=Path(config.label_pth),
            enc_model=Path(config.enc_model),
            params_batchSize=int(self.params.BATCH_SIZE),
            params_outputSize=int(self.params.OUTPUT_SIZE),
            params_epoch=int(self.params.EPOCHS),
            params_lr=float(self.params.LR),
            optmi_wd=float(self.params.WEIGHT_DECAY),
            scheduler_factor=float(self.params.factor),
            scheduler_patience=int(self.params.patience),
            scheduler_min_lr=float(self.params.min_lr),
        )

        return model_train_config
    
    def get_test_config(self) -> TestConfig:
        config = self.config.model_test

        create_directories([config.root_dir])

        test_config = TestConfig(
            root_dir= Path(config.root_dir),
            test_pth= Path(config.test_pth),
            model_pth= Path(config.model_pth),
            val_images_dir= Path(config.val_images_dir),
            val_labels_dir= Path(config.val_labels_dir)
            
        )
        return test_config
