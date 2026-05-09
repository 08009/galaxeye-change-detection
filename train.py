
import os, sys, random, yaml, csv
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
import segmentation_models_pytorch as smp

sys.path.insert(0, '/content/drive/MyDrive/galaxeye_project')
from dataset import ChangeDetectionDataset
from model import build_model

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

class DiceBCELoss(nn.Module):
    def __init__(self, pos_weight):
        super().__init__()
        self.bce  = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([pos_weight]))
        self.dice = smp.losses.DiceLoss(mode="binary")

    def forward(self, pred, target):
        # pred:   (B, 1, H, W)
        # target: (B, H, W) -> unsqueeze to (B, 1, H, W)
        target_4d = target.unsqueeze(1)
        bce_loss  = self.bce(pred.squeeze(1), target)
        dice_loss = self.dice(pred, target_4d)
        return 0.5 * bce_loss + 0.5 * dice_loss

def compute_metrics(preds_flat, targets_flat):
    TP = int(((preds_flat == 1) & (targets_flat == 1)).sum())
    FP = int(((preds_flat == 1) & (targets_flat == 0)).sum())
    FN = int(((preds_flat == 0) & (targets_flat == 1)).sum())
    precision = TP / (TP + FP + 1e-8)
    recall    = TP / (TP + FN + 1e-8)
    f1        = 2 * precision * recall / (precision + recall + 1e-8)
    iou       = TP / (TP + FP + FN + 1e-8)
    return {"precision": round(precision,4),
            "recall":    round(recall,4),
            "f1":        round(f1,4),
            "iou":       round(iou,4)}

def train(config_path):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    set_seed(cfg["seed"])
    os.makedirs(cfg["paths"]["checkpoint_dir"], exist_ok=True)

    p = cfg["paths"]

    train_ds = ChangeDetectionDataset(
        p["train_pre"], p["train_post"], p["train_target"],
        split="train", image_size=cfg["image_size"])

    val_ds = ChangeDetectionDataset(
        p["val_pre"], p["val_post"], p["val_target"],
        split="val", image_size=cfg["image_size"])

    train_dl = DataLoader(train_ds, batch_size=cfg["batch_size"],
                          shuffle=True,  num_workers=cfg["num_workers"],
                          pin_memory=True)
    val_dl   = DataLoader(val_ds,   batch_size=cfg["batch_size"],
                          shuffle=False, num_workers=cfg["num_workers"],
                          pin_memory=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nDevice: {device}")
    print(f"Train: {len(train_ds)} samples | Val: {len(val_ds)} samples\n")

    model     = build_model(cfg["encoder"], cfg["encoder_weights"]).to(device)
    loss_fn   = DiceBCELoss(pos_weight=cfg["pos_weight"]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=cfg["learning_rate"],
                                  weight_decay=cfg["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, T_max=cfg["epochs"])
    scaler    = GradScaler()

    best_f1  = 0.0
    log_rows = []

    for epoch in range(1, cfg["epochs"] + 1):

        # ── TRAIN LOOP ──────────────────────────────────────────
        model.train()
        train_loss = 0.0
        for x, mask in train_dl:
            x, mask = x.to(device), mask.to(device)
            optimizer.zero_grad()
            with autocast():
                pred = model(x)          # (B, 1, H, W)
                loss = loss_fn(pred, mask)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_loss += loss.item()
        train_loss /= len(train_dl)

        # ── VAL LOOP ────────────────────────────────────────────
        model.eval()
        all_preds, all_targets = [], []
        with torch.no_grad():
            for x, mask in val_dl:
                x = x.to(device)
                pred = torch.sigmoid(model(x))   # (B,1,H,W)
                pred_bin = (pred.squeeze(1).cpu().numpy() > 0.5).astype(np.uint8)
                all_preds.append(pred_bin.flatten())
                all_targets.append(mask.numpy().astype(np.uint8).flatten())

        all_preds   = np.concatenate(all_preds)
        all_targets = np.concatenate(all_targets)
        metrics     = compute_metrics(all_preds, all_targets)

        scheduler.step()

        print(f"Epoch {epoch:03d}/{cfg['epochs']} | "
              f"Loss: {train_loss:.4f} | "
              f"F1: {metrics['f1']:.4f} | "
              f"IoU: {metrics['iou']:.4f} | "
              f"Prec: {metrics['precision']:.4f} | "
              f"Rec: {metrics['recall']:.4f}")

        log_rows.append({"epoch": epoch,
                         "train_loss": round(train_loss, 4),
                         **metrics})

        # save best checkpoint
        if metrics["f1"] > best_f1:
            best_f1   = metrics["f1"]
            ckpt_path = os.path.join(cfg["paths"]["checkpoint_dir"],
                                     "best_model.pth")
            torch.save(model.state_dict(), ckpt_path)
            print(f"  ✅ Best model saved — Val F1: {best_f1:.4f}")

    # save training log CSV
    log_path = os.path.join(cfg["paths"]["checkpoint_dir"], "train_log.csv")
    with open(log_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=log_rows[0].keys())
        writer.writeheader()
        writer.writerows(log_rows)

    print(f"\nTraining complete. Best Val F1: {best_f1:.4f}")
    print(f"Checkpoint saved at: {ckpt_path}")

