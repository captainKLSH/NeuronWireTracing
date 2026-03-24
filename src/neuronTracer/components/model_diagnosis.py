from src.neuronTracer.entity.config import ModelDiagnosisConfig
from src.neuronTracer import logger
import torch
import multiprocessing
import os
class modelDiagnostic:
    def __init__(self, config:ModelDiagnosisConfig):
        self.config=config
        self.image_dir = config.train_img
        self.label_dir = config.label_pth
        self.img_files = sorted(self.image_dir.glob('box_*.pt'))
        self.lbl_files = sorted(self.label_dir.glob('box_*.pt'))
        
        

    def verifyPair(self):
        """Verify pairing
        You will be prompted twice — once for image files, once for label files.
        """
        img_files = self.img_files
        lbl_files = self.lbl_files

        logger.info(f'\nImage files : {[f.name for f in img_files]}')
        logger.info(f'\nLabel files : {[f.name for f in lbl_files]}')

        assert len(img_files) > 0,           'No image files found — re-run this cell'
        assert len(img_files) == len(lbl_files), 'Image / label count mismatch'
        for i, l in zip(img_files, lbl_files):
            assert i.name == l.name, logger.info(f'Filename mismatch: {i.name} vs {l.name}')

        logger.info(f'All {len(img_files)} pairs matched ✅')
    
    def sysConfig(self):
        logger.info("🔍 --- Project Diagnostic ---")
        device = torch.device("cpu")

        # 1. Multiprocessing Check
        # Colab uses Linux, which defaults to 'fork'. 
        # For CUDA, 'spawn' is technically safer to avoid deadlocks.
        logger.info("[MULTIPROCESSING]")
        try:
            start_method = multiprocessing.get_start_method()
            logger.info(f"  Start method : {start_method}")
            cpu_count = multiprocessing.cpu_count()
            logger.info(f"  CPU cores    : {cpu_count} logical")
        except Exception as e:
            logger.error(f"  Multiprocessing check failed: {e}")

        # 2. Hardware & Backend Check
        logger.info("[DEVICE & BACKEND]")
        
        # NVIDIA CUDA (Colab Standard)
        if torch.cuda.is_available():
            device = torch.device("cuda")
            prop = torch.cuda.get_device_properties(0)
            
            logger.info(f"  Backend      ⚙️ : CUDA (NVIDIA)")
            logger.info(f"  GPU Model    : {prop.name}")
            logger.info(f"  VRAM Total   : {prop.total_memory / 1e9:.1f} GB")
            
            # Check for 'Compute Capability' (DINOv2 runs best on 7.0+)
            cc = f"{prop.major}.{prop.minor}"
            logger.info(f"  Compute Cap  : {cc}")
            
            # Check Memory Fragmentation
            vram_reserved = torch.cuda.memory_reserved(0) / 1e9
            vram_allocated = torch.cuda.memory_allocated(0) / 1e9
            logger.info(f"  VRAM Reserved: {vram_reserved:.1f} GB")
            logger.info(f"  VRAM Active  : {vram_allocated:.1f} GB")

            # Smoke Test
            try:
                test_tensor = torch.zeros((100, 100), device=device)
                logger.info("  Smoke test 💨 : CUDA Tensor creation OK")
                del test_tensor # Clean up immediately
            except Exception as e:
                logger.warning(f"  Smoke test 💨 : FAILED — {e}")

        # Apple Silicon (For when you run locally)
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
            logger.info("  Backend      ⚙️ : Apple Silicon (MPS)")
            logger.info("  Status       ✅: OK")

        else:
            logger.warning("  Backend      ⚙️ : CPU only — No Accelerator Found")

        # 3. Environment Context (Colab Specific)
        if 'COLAB_GPU' in os.environ:
            logger.info("  Environment  🌐: Google Colab detected")
        
        return device
    
    def dataInspect(self):
        '''
        Confirms the shape, dtype, and value range before anything is built.
        '''
        sample_img = torch.load(self.img_files[0], weights_only=True)
        sample_lbl = torch.load(self.lbl_files[0], weights_only=True)

        logger.info('=== Image (box_0000.pt from images/) ===')
        logger.info(f'  shape  : {sample_img.shape}')    # expect [1, Z, H, W] or [Z, H, W]
        logger.info(f'  dtype  : {sample_img.dtype}')
        logger.info(f'  range  : [{sample_img.min():.3f}, {sample_img.max():.3f}]')

        logger.info('=== Label (box_0000.pt from labels/) ===')
        logger.info(f'  shape  : {sample_lbl.shape}')
        logger.info(f'  dtype  : {sample_lbl.dtype}')
        logger.info(f'\n unique : {sample_lbl.unique().tolist()}')

        # DINOv2 operates on 2D slices — we extract one Z-slice per sample
        # The spatial dims (H, W) get resized to 224×224 inside DinoAdapter
        SPATIAL_Z = sample_img.shape[-3] if sample_img.dim() == 4 else sample_img.shape[-3]
        logger.info(f'Z-depth per volume : {SPATIAL_Z}')
        logger.info(f'Total 2D slices    : {len(self.img_files) * SPATIAL_Z}')

        