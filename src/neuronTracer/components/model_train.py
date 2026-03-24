import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm.auto import tqdm
import os # Import os module for file operations
from src.neuronTracer import logger
from src.neuronTracer.entity.config import ModelTrainConfig


class ModelTraining:
    def __init__(self,config: ModelTrainConfig,encoder:nn.Module, device: torch.device, train_loader, val_loader ):
        self.config =config
        self.epochs= config.params_epoch
        self.device = device
        self.lr= config.params_lr
        self.model=encoder
        self.weights=config.enc_model
        self.WD=config.optmi_wd
        self.fact=config.scheduler_factor
        self.pat=config.scheduler_patience
        self.minlr=config.scheduler_min_lr
        self.Tload = train_loader
        self.Vload = val_loader

    # ── Loss function ─────────────────────────────────────────────────────────────
    # FIX: original used nn.BCELoss() on sigmoid output → NaN risk + 0.59 plateau
    # Dice+BCE is class-balanced: the 1% foreground voxels get equal weight to
    # the 99% background, so the model cannot cheat by predicting all-zero.
    def loss_function(self,predictions, targets, smooth=1.0):
        """
        predictions : raw logits [B, 1, 224, 224]  (no sigmoid applied yet)
        targets     : float {0,1} [B, 1, 224, 224]
        """
        # BCE term — sigmoid applied internally (numerically stable)Dynamic Class Balancing penalty
        bg    = (1 - targets).sum().clamp(min=1)
        wall = targets.sum().clamp(min=1)
        weight_penalty  = (bg / wall).clamp(1, 50).to(self.device)

        bce  = F.binary_cross_entropy_with_logits(predictions, targets,pos_weight=weight_penalty)

        # Dice term — balance foreground vs background
        prob = torch.sigmoid(predictions)
        p    = prob.reshape(-1)
        t    = targets.reshape(-1)
        dice = 1.0 - (2.0*(p*t).sum() + smooth) / (p.sum() + t.sum() + smooth)

        return 0.5 * bce + 0.6 * dice #50% about pixel-perfect math (BCE) and 60% about drawing the right shapes (Dice)
    
    def training(self):
        logger.info('Initialize our Hybrid Model')
        self.model.to(self.device)
        optimizer = AdamW(self.model.botanist_decoder.parameters(), lr=self.lr, weight_decay=self.WD)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=self.fact, patience=self.pat, min_lr=self.minlr
        )
        best_dice = 0.0
        best_path = self.weights
        if best_path.exists() and os.path.getsize(best_path) == 0:
            logger.info(f'Empty checkpoint found at {best_path} — deleting.')
            os.remove(best_path)

        if best_path.exists():
            logger.info(f'Loading previous weights from {best_path}...')
            try:
                self.model.botanist_decoder.load_state_dict(
                    torch.load(best_path, map_location=self.device, weights_only=True)
                )
                logger.info('Weights loaded OK.')
            except RuntimeError as e:
                logger.info(f'Load failed: {e} — starting from scratch.')
                os.remove(best_path)
        else:
            logger.info('No checkpoint found — starting from scratch.')

        logger.info('\nStarting fine-tuning loop...')

        for epoch in range(1, self.epochs + 1):

            # ── Train ──────────────────────────────────────────────────────────────────
            self.model.train()
            self.model.dino_encoder.eval()   # DINOv2 stays frozen always
            tr_loss = 0.0

            loop = tqdm(self.Tload,
                        desc=f'Epoch {epoch}/{self.epochs} [train]',
                        leave=False)

            for batch_imgs, batch_labels in loop:
                batch_imgs   = batch_imgs.to(self.device).float()
                batch_labels = batch_labels.to(self.device).float()
                
                optimizer.zero_grad()

                # Forward pass
                predictions = self.model(batch_imgs)              # [B, 1, 224, 224] logits

                # Grade it
                loss = self.loss_function(predictions, batch_labels)

                # Update ONLY the decoder
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.botanist_decoder.parameters(), 1.0)
                optimizer.step()

                tr_loss += loss.item()
                loop.set_postfix(loss=f'{loss.item():.4f}')

            # ── Validate ───────────────────────────────────────────────────────────────
            self.model.eval()
            val_loss = val_dice = 0.0
            all_preds, all_labels = [], []
            with torch.no_grad():
                for batch_imgs, batch_labels in self.Vload:
                    batch_imgs   = batch_imgs.to(self.device).float()
                    batch_labels = batch_labels.to(self.device).float()

                    
                    predictions  = self.model(batch_imgs)
                    val_loss    += self.loss_function(predictions, batch_labels).item()

                    # Dice score on binary threshold
                    prob  = torch.sigmoid(predictions)
                    all_preds.append(prob.cpu())
                    all_labels.append(batch_labels.cpu())

            all_preds  = torch.cat(all_preds)
            all_labels = torch.cat(all_labels)
            best_t, best_epoch_dice = 0.5, 0.0
            for t in torch.arange(0.1, 0.7, 0.05):
                preds = (all_preds > t).float()
                tp    = (preds * all_labels).sum()
                d     = (2*tp + 1) / (preds.sum() + all_labels.sum() + 1)
                if d.item() > best_epoch_dice:
                    best_epoch_dice = d.item()
                    best_t = t.item()
            avg_tr   = tr_loss  / len(self.Tload)
            avg_val  = val_loss / len(self.Vload)

            scheduler.step(best_epoch_dice)

            # Same print format as original script + val metrics
            print(
                f'Epoch {epoch:02d} | '
                f'Train Loss: {avg_tr:.4f} | '
                f'Val Loss: {avg_val:.4f} | '
                f'Dice: {best_epoch_dice:.4f} | '
                f'Threshold: {best_t:.2f} | '
                f'LR: {optimizer.param_groups[0]["lr"]:.2e}'
            )

            if best_epoch_dice > best_dice:
                best_dice = best_epoch_dice
                torch.save(self.model.botanist_decoder.state_dict(), best_path)
                print(f'  New best Dice {best_dice:.4f} → saved to {best_path}')

        print('\nTraining complete! Saving your custom Botanist Decoder...')
        torch.save(self.model.botanist_decoder.state_dict(), self.config.root_dir)
        print(f'Saved: {self.config.root_dir}/dino_decoder_best.pth')
