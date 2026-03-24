import torch
import torch.nn as nn
import numpy as np
import zarr
import tifffile
import os
import torch.nn.functional as F
from tqdm import tqdm
from src.neuronTracer import logger
from src.neuronTracer.entity.config import TestConfig


# ─────────────────────────────────────────────────────────────────────────────
# 1. Rebuild the Advanced Architecture (Matches your training script perfectly)
# ─────────────────────────────────────────────────────────────────────────────

class DinoAdapter(nn.Module):
    def __init__(self):
        super().__init__()
        self.resize = nn.Upsample(size=(224, 224), mode='bilinear', align_corners=False)

    def forward(self, x):
        x = self.resize(x)         # [B, 1, 224, 224]
        x = x.repeat(1, 3, 1, 1)   # [B, 3, 224, 224]
        return x

class BotanistDecoder(nn.Module):
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
            # NO Sigmoid here! We output raw logits.
        )

    def forward(self, patch_tokens):
        # patch_tokens: [B, 256, 768]
        B = patch_tokens.shape[0]
        x = patch_tokens.transpose(1, 2).reshape(B, self.dino_dim, self.PATCH_GRID, self.PATCH_GRID)
        x = self.deconv1(x)   # [B, 256,  64,  64]
        x = self.deconv2(x)   # [B,  64, 128, 128]
        x = self.deconv3(x)   # [B,  32, 256, 256]
        return self.final(x)  # [B,   1, 224, 224] raw logits

class HybridDinoTracker(nn.Module):
    def __init__(self):
        super().__init__()
        self.adapter = DinoAdapter()
        logger.info("Waking up Meta's DINOv2 Satellite...")
        self.dino_encoder = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14', trust_repo=True)
        
        # We don't need to freeze here since we use torch.no_grad() in inference anyway
        self.botanist_decoder = BotanistDecoder()

    def forward(self, x):
        x = self.adapter(x)
        # Pull the spatial patch grid instead of the flat CLS token
        with torch.no_grad():
            features = self.dino_encoder.get_intermediate_layers(x, n=1)
        patch_tokens = features[0]
        return self.botanist_decoder(patch_tokens)

# ─────────────────────────────────────────────────────────────────────────────
# 2. The Test Taker
# ─────────────────────────────────────────────────────────────────────────────

class Test:
    def __init__(self, config: TestConfig):
        self.config=config
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        self.model = HybridDinoTracker().to(self.device)
        self.weights_path = self.config.model_pth

    def take_final(self):
        logger.info("Preparing for the Final Exam on Apple Silicon...")
        
        
        if os.path.exists(self.weights_path):
            self.model.botanist_decoder.load_state_dict(torch.load(self.weights_path, map_location=self.device, weights_only=True))
            logger.info("✅ Successfully loaded your highly trained custom brain!")
        else:
            logger.info(f"❌ ERROR: Could not find {self.weights_path}.")
            return
            
        self.model.eval()
        
        logger.info("Opening the blank test dataset...")
        test_zarr = zarr.open(self.config.test_pth, mode='r')
        
        final_3d_canvas = np.zeros(test_zarr.shape, dtype=np.float32)
        Z_depth = test_zarr.shape[0]
        box_size = 64 
        
        logger.info("The AI is now tracing the final exam. Please wait...")
        
        with torch.no_grad(): 
            for z in range(Z_depth):
                for y in range(0, test_zarr.shape[1], box_size):
                    for x in range(0, test_zarr.shape[2], box_size):
                        
                        test_chunk = test_zarr[z, y:y+box_size, x:x+box_size]
                        
                        if test_chunk.shape[0] != box_size or test_chunk.shape[1] != box_size:
                            continue
                            
                        input_tensor = torch.from_numpy(test_chunk).float().unsqueeze(0).unsqueeze(0).to(self.device)
                        
                        # 1. The AI draws the lines (Outputs 224x224 raw logits)
                        raw_logits = self.model(input_tensor)
                        
                        # 2. Convert to percentages
                        ai_tracing_224 = torch.sigmoid(raw_logits)
                        
                        # 🔥 THE FIX: Shrink the 224x224 drawing back down to 64x64
                        ai_tracing_64 = F.interpolate(
                            ai_tracing_224, 
                            size=(box_size, box_size), 
                            mode='bilinear', 
                            align_corners=False
                        )
                        
                        # 3. Paste the properly sized drawing onto our massive billboard
                        final_3d_canvas[z, y:y+box_size, x:x+box_size] = ai_tracing_64.squeeze().cpu().numpy()
                        
                if z % 10 == 0:
                    logger.info(f"Traced slice {z} / {Z_depth}...")

        logger.info("🎉 Exam finished! Saving the completed 3D tracing...")
        output_path = os.path.join(self.config.root_dir, "final.tif")
        tifffile.imwrite(output_path, final_3d_canvas)
        logger.info("SUCCESS! You can now upload 'final.tif' to the SNEMI3D Leaderboard.")

    def run_official_evaluation(self):
        logger.info("Setting up the Validation Grader on Apple Silicon...")
        
        
        if os.path.exists(self.weights_path):
            self.model.botanist_decoder.load_state_dict(torch.load(self.weights_path, map_location=self.device, weights_only=True))
            logger.info("✅ Botanist brain loaded successfully!")
        else:
            logger.info(f"❌ Could not find {self.weights_path}")
            return
            
        self.model.eval()
        
        # 2. Point to the NEW validation folders
        val_images_dir = self.config.val_images_dir
        val_labels_dir = self.config.val_labels_dir
        
        val_files = sorted([f for f in os.listdir(val_images_dir) if f.endswith('.pt')])
        logger.info(f"Found {len(val_files)} isolated validation files. Beginning grading...")
        
        # Trackers for our biological metrics
        total_tp = 0
        total_fp = 0
        total_fn = 0
        
        with torch.no_grad():
            for file in tqdm(val_files, desc="Grading AI"):
                # Load the raw .pt files directly into memory
                img_tensor = torch.load(os.path.join(val_images_dir, file), map_location=self.device, weights_only=True)
                lbl_tensor = torch.load(os.path.join(val_labels_dir, file), map_location='cpu', weights_only=True).numpy()
                
                # 🔥 THE 21-CHANNEL FIX: Flip [1, 7, 64, 64] to [7, 1, 64, 64]
                img_tensor = img_tensor.transpose(0, 1)
                
                # 1. AI makes a prediction on all 7 slices simultaneously
                raw_logits = self.model(img_tensor)
                ai_tracing_224 = torch.sigmoid(raw_logits)
                
                # 2. Shrink the drawing back to 64x64 to match the answer key
                original_size = lbl_tensor.shape[-1] 
                ai_tracing_resized = F.interpolate(
                    ai_tracing_224, 
                    size=(original_size, original_size), 
                    mode='bilinear', 
                    align_corners=False
                )
                
                # 🔥 FLIP IT BACK: [7, 1, 64, 64] back to [1, 7, 64, 64] to match the label shape
                ai_tracing_final = ai_tracing_resized.transpose(0, 1).cpu().numpy()
                
                # 3. Convert to Hard Yes/No decisions (Threshold at 50% confidence)
                preds_binary = (ai_tracing_final > 0.5).astype(bool)
                truth_binary = (lbl_tensor > 0).astype(bool)
                
                # 4. Calculate Confusion Matrix Metrics on the fly using high-speed NumPy
                total_tp += np.sum(preds_binary & truth_binary)
                total_fp += np.sum(preds_binary & ~truth_binary)
                total_fn += np.sum(~preds_binary & truth_binary)
                
        # Calculate Final Mathematics
        precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
        recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
        dice_score = (2 * total_tp) / (2 * total_tp + total_fp + total_fn) if (2 * total_tp + total_fp + total_fn) > 0 else 0.0
        iou = total_tp / (total_tp + total_fp + total_fn) if (total_tp + total_fp + total_fn) > 0 else 0.0
        
        logger.info("\n" + "="*40)
        logger.info("📊 UNBIASED VALIDATION REPORT")
        logger.info("="*40)
        logger.info(f"Precision (Hallucination Check):  {precision * 100:.2f}%")
        logger.info(f"Recall    (Blind-Spot Check):     {recall * 100:.2f}%")
        logger.info(f"Dice / F1 (Official Accuracy):    {dice_score * 100:.2f}%")
        logger.info(f"IoU       (Geometric Overlap):    {iou * 100:.2f}%")
        logger.info("="*40)
