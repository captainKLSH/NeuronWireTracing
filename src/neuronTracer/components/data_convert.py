import torch
import os
import zarr
from src.neuronTracer.entity.config import TorchConvertConfig
from src.neuronTracer import logger
class TorchConvert:
    def __init__(self, config:TorchConvertConfig):
        self.config=config

    def extract_and_store_tensors(self):
        """
        Slices the Zarr warehouse into PyTorch Tensors with boundary safety.
        """
        images = zarr.open(self.config.train_data, mode='r')
        labels = zarr.open(self.config.label_data, mode='r')

        img_folder = os.path.join(self.config.root_dir, "images")
        lbl_folder = os.path.join(self.config.root_dir, "labels")
        os.makedirs(img_folder, exist_ok=True)
        os.makedirs(lbl_folder, exist_ok=True)

        # Get our (7, 64, 64) dimensions
        z_step, y_step, x_step = self.config.params_box_size
        z_max, y_max, x_max = images.shape

        counter = 0
        # The Triple Loop: The standard way to slice 3D volumes
        for z in range(0, z_max - z_step + 1, z_step):
            for y in range(0, y_max - y_step + 1, y_step):
                for x in range(0, x_max - x_step + 1, x_step):
                    
                    # Pull and convert
                    img_chunk = images[z:z+z_step, y:y+y_step, x:x+x_step]
                    lbl_chunk = labels[z:z+z_step, y:y+y_step, x:x+x_step]

                    # Convert to FloatTensor and add (C, Z, H, W) dimension
                    # PyTorch 3D CNNs expect 4D input: (Channel, Depth, Height, Width)
                    img_tensor = torch.from_numpy(img_chunk).float().unsqueeze(0)
                    lbl_tensor = torch.from_numpy(lbl_chunk).float().unsqueeze(0)

                    # Save using a padded index (e.g., box_0001.pt) for better folder sorting
                    torch.save(img_tensor, os.path.join(img_folder, f"box_{counter:04d}.pt"))
                    torch.save(lbl_tensor, os.path.join(lbl_folder, f"box_{counter:04d}.pt"))
                    
                    counter += 1

        logger.info(f"Successfully stored {counter} tensor pairs in {self.config.root_dir}")