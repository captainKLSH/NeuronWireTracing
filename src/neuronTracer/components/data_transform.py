import os
import tifffile
import zarr
from src.neuronTracer import logger
from pathlib import Path
from src.neuronTracer.entity.config import DataTransformConfig
class DataTransform:
    def __init__(self, config:DataTransformConfig):
        self.config=config
    
    def repackage_tiff(self):
        """
        Opens a 3D TIFF image stack and repackages it into small Zarr chunks.
        box_size: e.g., (64, 64, 64) for perfect cube chunks
        """
        try:
            logger.info(f"Loading 3D Image Stack from: {self.config.train_data}")
            train_block = tifffile.imread(self.config.train_data)
            
            # Create a specific sub-folder for images
            train_path = os.path.join(self.config.root_dir, "train_images.zarr")
            
            zarr.array(
                train_block, 
                chunks=self.config.params_box_size,
                store=train_path,
                overwrite=True
            )
            logger.info(f"Images chunked into {self.config.params_box_size} and saved to {train_path}")
            # Free up RAM before loading the next big block
            del train_block
            # 2. Process the Labels (Masks)
            logger.info(f"Loading 3D Label Stack from: {self.config.label_data}")
            label_block = tifffile.imread(self.config.label_data)
            
            # Create a specific sub-folder for labels
            label_path = os.path.join(self.config.root_dir, "train_labels.zarr")
            
            zarr.array(
                label_block,
                store=label_path, 
                chunks=self.config.params_box_size,
                overwrite=True
            )
            logger.info(f"Labels chunked into {self.config.params_box_size} and saved to {label_path}")
            
            del label_block
            # 3. Process the Test 
            logger.info(f"Loading 3D Label Stack from: {self.config.test_data}")
            test_block = tifffile.imread(self.config.test_data)
            
            # Create a specific sub-folder for labels
            test_path = os.path.join(self.config.root_dir, "test.zarr")
            
            zarr.array(
                test_block,
                store=test_path, 
                chunks=self.config.params_box_size,
                overwrite=True
            )
            logger.info(f"test chunked into {self.config.params_box_size} and saved to {test_path}")
            
            del test_block
            logger.info("Transformation Complete: Data is now ready for the 3D CNN pipeline.")

        except Exception as e:
            logger.error(f"Error during data transformation: {e}")
            raise e


