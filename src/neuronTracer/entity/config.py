from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir: Path
    source_URL: str
    local_data_file: Path
    unzip_dir: Path

@dataclass(frozen=True)
class DataTransformConfig:
    root_dir: Path
    train_data: Path
    label_data:Path
    test_data: Path
    params_box_size: list

@dataclass(frozen=True)
class TorchConvertConfig:
    root_dir: Path
    train_data: Path
    label_data:Path
    params_box_size: list


@dataclass(frozen=True)
class ModelDiagnosisConfig:
    train_img: Path
    label_pth:Path
    params_box_size: list

@dataclass(frozen=True)
class ModelBuildEncConfig:
    train_img: Path
    label_pth:Path
    params_box_size: list
    params_batchSize: int
    params_outputSize: int
    params_epoch: int
    params_lr: float
    params_augmentT: bool
    params_augmentL:bool 

@dataclass(frozen=True)
class ModelTrainConfig:
    root_dir: Path
    train_img: Path
    label_pth:Path
    enc_model: Path
    params_batchSize: int
    params_outputSize: int
    params_epoch: int
    params_lr: float
    optmi_wd: float
    scheduler_factor: float
    scheduler_patience: int
    scheduler_min_lr: float

@dataclass(frozen=True)
class TestConfig:
    root_dir: Path
    test_pth: Path
    model_pth: Path
    val_images_dir: Path
    val_labels_dir: Path