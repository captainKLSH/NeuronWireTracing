import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm
from src.neuronTracer import logger

class Adapter(nn.Module):
    """
    1. The Adapter (same name as original)
        Converts [B, 1, H, W] grayscale → [B, 3, 224, 224] fake-RGB for DINOv2
    """
    def __init__(self):
        super().__init__()
        self.resize = nn.Upsample(size=(224, 224), mode='bilinear', align_corners=False)

    def forward(self, x):
        x = self.resize(x)         # [B, 1, 224, 224]
        x = x.repeat(1, 3, 1, 1)  # [B, 3, 224, 224]  — fake RGB
        return x

class BTDecoder(nn.Module):
    '''
    # ─────────────────────────────────────────────────────────────────────────────
    # 2. The Botanist Decoder 
    #
    # FIX: original took flat CLS token [B, 768] → view(B,768,1,1) — 1x1 spatial!
    # Now takes 256 patch tokens [B, 256, 768] → reshape [B, 768, 16, 16] — real map.
    #
    # FIX: Sigmoid removed. Output is raw logits.
    #      Use BCEWithLogitsLoss (stable). Apply sigmoid only at inference.
    # ─────────────────────────────────────────────────────────────────────────────
    '''
    PATCH_GRID = 16   # 224 // 14 = 16 patches per side
    def __init__(self):
        super().__init__()
        self.dino_dim = 768

        self.deconv1 = nn.Sequential(
            nn.ConvTranspose2d(768, 256, kernel_size=4, stride=4),   # 16 → 64
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        self.deconv2 = nn.Sequential(
            nn.ConvTranspose2d(256, 64, kernel_size=2, stride=2),    # 64 → 128
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.deconv3 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),     # 128 → 256
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        self.final = nn.Sequential(
            nn.Conv2d(32, 1, kernel_size=1),
            nn.Upsample(size=(224, 224), mode='bilinear', align_corners=False),
            # NO Sigmoid — BCEWithLogitsLoss fuses it stably
        )

    def forward(self, patch_tokens):
        # patch_tokens: [B, 256, 768]
        B = patch_tokens.shape[0]
        x = patch_tokens.transpose(1, 2).reshape(B, self.dino_dim,
                                                  self.PATCH_GRID, self.PATCH_GRID)
        x = self.deconv1(x)   # [B, 256,  64,  64]
        x = self.deconv2(x)   # [B,  64, 128, 128]
        x = self.deconv3(x)   # [B,  32, 256, 256]
        return self.final(x)  # [B,   1, 224, 224]  raw logits

class DinoEncoder(nn.Module):
    """
    # ─────────────────────────────────────────────────────────────────────────────
    # 3. The Master Model (same name as original)
    # Satellite Frozen DINOv2 brain + unfrozen BotanistDecoder
    # ─────────────────────────────────────────────────────────────────────────────
    """
    def __init__(self):
        super().__init__()
        self.adapter = Adapter()

        logger.info('Downloading DINOv2-ViT-B/14 from Meta (~330 MB, once) ...')
        self.dino_encoder = torch.hub.load(
            'facebookresearch/dinov2',
            'dinov2_vitb14',
            trust_repo=True,
        )

        # Freeze the entire DINOv2 brain
        for param in self.dino_encoder.parameters():
            param.requires_grad = False
        self.dino_encoder.eval()
        logger.info(f'  DINOv2 frozen ({sum(p.numel() for p in self.dino_encoder.parameters())/1e6:.0f}M params)')

        self.botanist_decoder = BTDecoder()
        logger.info(f'  Decoder trainable ({sum(p.numel() for p in self.botanist_decoder.parameters())/1e6:.1f}M params)')

    def forward(self, x):
        # x: [B, 1, H, W]  grayscale EM slice
        x = self.adapter(x)                                    # [B, 3, 224, 224]

        # FIX: use patch tokens (256 spatial tokens), NOT the CLS token
        with torch.no_grad():
            features = self.dino_encoder.get_intermediate_layers(x, n=1)
        patch_tokens = features[0]                             # [B, 256, 768]

        return self.botanist_decoder(patch_tokens)             # [B, 1, 224, 224]

# ── 1. The Dataset Class ──────────────────────────────────────────────────────
class BoxPtDataset(Dataset):
    """
    Reads box_NNNN.pt pairs. Expects direct lists of files so we can easily 
    split them into Train and Validation sets later.
    """
    def __init__(self, img_files, lbl_files, out_size=224, augment=False):
        self.samples = []
        self.img_files = img_files
        self.lbl_files = lbl_files
        self.out_size = out_size
        self.augment = augment

        # Index every Z-slice from every volume
        for img_path, lbl_path in zip(self.img_files, self.lbl_files):
            # We use weights_only=True for security, as recommended by PyTorch
            vol = torch.load(img_path, map_location='cpu', weights_only=True)
            if vol.dim() == 3:   
                vol = vol.unsqueeze(0)
            
            n_z = vol.shape[1]
            for z in range(n_z):
                self.samples.append((img_path, lbl_path, z))

        logger.info(f'BoxPtDataset initialized: {len(self.samples)} total slices from {len(self.img_files)} volumes.')

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, lbl_path, z = self.samples[idx]

        img = torch.load(img_path, map_location='cpu', weights_only=True).float()
        lbl = torch.load(lbl_path, map_location='cpu', weights_only=True).float()

        # Ensure [1, Z, H, W]
        if img.dim() == 3: img = img.unsqueeze(0)
        if lbl.dim() == 3: lbl = lbl.unsqueeze(0)

        img_slice = img[:, z, :, :]   # [1, H, W]
        lbl_slice = lbl[:, z, :, :]   # [1, H, W]

        # 1. Normalize image to [0, 1]
        mn, mx = img_slice.min(), img_slice.max()
        img_slice = (img_slice - mn) / (mx - mn + 1e-6)

        # 2. THE PHOTOCOPY FIX: Convert 401 unique instance IDs into binary cell walls {0, 1}
        # Assuming 1 is the cell boundary wall we want the AI to draw
        lbl_slice = (lbl_slice > 0).float()

        # 3. Resize label to 224x224 to match the DINOv2 / Decoder output
        lbl_slice = F.interpolate(
            lbl_slice.unsqueeze(0),
            size=(self.out_size, self.out_size),
            mode='nearest',
        ).squeeze(0)   # [1, 224, 224]

        # 4. Augmentation (training only)
        if self.augment:
            if torch.rand(1) > 0.5:
                img_slice = torch.flip(img_slice, [1])   # flip H
                lbl_slice = torch.flip(lbl_slice, [1])
            if torch.rand(1) > 0.5:
                img_slice = torch.flip(img_slice, [2])   # flip W
                lbl_slice = torch.flip(lbl_slice, [2])
            
            # Intensity jitter
            scale = 0.9 + 0.2 * torch.rand(1).item()
            img_slice = (img_slice * scale).clamp(0.0, 1.0)

        return img_slice, lbl_slice

# ── 2. The Dataloader Factory ─────────────────────────────────────────────────
def build_dataloaders(config):
    """
    Takes your ConfigurationManager config, splits the files, 
    and returns ready-to-use PyTorch DataLoaders.
    """
    logger.info('Locating files and building datasets...')
    
    # Grab the paths from the config
    image_dir = config.train_img
    label_dir = config.label_pth  
    
    # Safely get all files (Fixed the label_dir typo here)
    img_files = sorted(image_dir.glob('box_*.pt'))
    lbl_files = sorted(label_dir.glob('box_*.pt'))
    
    assert len(img_files) == len(lbl_files), "Mismatch between number of images and labels!"
    assert len(img_files) > 1, "Need at least 2 files to create a train/val split!"

    # Split: all but last volume = train, last volume = val
    train_img_files = img_files[:-1]
    train_lbl_files = lbl_files[:-1]
    val_img_files   = img_files[-1:]
    val_lbl_files   = lbl_files[-1:]

    # Instantiate the Datasets using the fixed arguments
    train_ds = BoxPtDataset(
        img_files=train_img_files, 
        lbl_files=train_lbl_files, 
        out_size=config.params_outputSize,
        augment=config.params_augmentT
    )
    
    val_ds = BoxPtDataset(
        img_files=val_img_files,   
        lbl_files=val_lbl_files,  
        out_size=config.params_outputSize,
        augment=config.params_augmentL # Assuming this is meant to be False for validation
    )

    # Wrap them in DataLoaders
    train_loader = DataLoader(
        train_ds, 
        batch_size=config.params_batchSize, 
        shuffle=True,
        num_workers=0, 
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_ds,   
        batch_size=config.params_batchSize, 
        shuffle=False,
        num_workers=0
    )

    # Verification printout
    images, labels = next(iter(train_loader))
    logger.info(f'\n[Data Flow Verification]')
    logger.info(f'Image Batch Shape: {list(images.shape)} (expect: [{config.params_batchSize}, 1, 64, 64])')
    logger.info(f'Label Batch Shape: {list(labels.shape)} (expect: [{config.params_batchSize}, 1, 224, 224])')
    logger.info(f'Data Range: Min={images.min():.2f}, Max={images.max():.2f}')
    logger.info(f'Label unique values (Should only be 0.0 and 1.0): {labels.unique().tolist()}\n')

    return train_loader, val_loader