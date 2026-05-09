
import os, sys, yaml
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

sys.path.insert(0, '/content/drive/MyDrive/galaxeye_project')
from dataset import ChangeDetectionDataset
from model import build_model

def compute_metrics(preds, targets):
    TP = int(((preds==1)&(targets==1)).sum())
    FP = int(((preds==1)&(targets==0)).sum())
    FN = int(((preds==0)&(targets==1)).sum())
    precision = TP/(TP+FP+1e-8)
    recall    = TP/(TP+FN+1e-8)
    f1        = 2*precision*recall/(precision+recall+1e-8)
    iou       = TP/(TP+FP+FN+1e-8)
    return precision, recall, f1, iou

def evaluate(config_path, weights_path, split="test"):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    p = cfg["paths"]
    pre_dir    = p[f"{split}_pre"]
    post_dir   = p[f"{split}_post"]
    target_dir = p[f"{split}_target"]

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load model — encoder_weights=None because we load our own weights
    model = build_model(cfg["encoder"], encoder_weights=None).to(device)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()
    print(f"Model loaded from: {weights_path}")

    ds = ChangeDetectionDataset(pre_dir, post_dir, target_dir,
                                split=split,
                                image_size=cfg["image_size"])
    dl = DataLoader(ds, batch_size=2, shuffle=False, num_workers=2)

    all_preds, all_targets = [], []
    sample_visuals = []   # store first 5 for plots

    with torch.no_grad():
        for i, (x, mask) in enumerate(dl):
            x = x.to(device)
            pred     = torch.sigmoid(model(x)).squeeze(1).cpu().numpy()
            pred_bin = (pred > 0.5).astype(np.uint8)

            all_preds.append(pred_bin.flatten())
            all_targets.append(mask.numpy().astype(np.uint8).flatten())

            if len(sample_visuals) < 5:
                sample_visuals.append((
                    x[0].cpu().numpy(),   # (4,H,W)
                    mask[0].numpy(),      # (H,W)
                    pred_bin[0]           # (H,W)
                ))

    all_preds   = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)

    prec, rec, f1, iou = compute_metrics(all_preds, all_targets)

    print(f"\n{'='*40}")
    print(f"  Results on [{split}] split")
    print(f"{'='*40}")
    print(f"  IoU       : {iou:.4f}")
    print(f"  Precision : {prec:.4f}")
    print(f"  Recall    : {rec:.4f}")
    print(f"  F1 Score  : {f1:.4f}")
    print(f"{'='*40}\n")

    # ── Confusion Matrix ───────────────────────────────────────
    cm   = confusion_matrix(all_targets, all_preds)
    disp = ConfusionMatrixDisplay(cm, display_labels=["No-Change","Change"])
    disp.plot(cmap="Blues")
    plt.title(f"Confusion Matrix — {split}")
    out_dir = '/content/drive/MyDrive/galaxeye_project'
    plt.savefig(f"{out_dir}/confusion_matrix_{split}.png",
                dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Confusion matrix saved ✅")

    # ── Qualitative Visualisations ─────────────────────────────
    fig, axes = plt.subplots(5, 3, figsize=(12, 20))
    for i, (x_np, gt, pred) in enumerate(sample_visuals):
        eo_img  = x_np[:3].transpose(1,2,0)   # (H,W,3)
        eo_img  = (eo_img - eo_img.min()) / (eo_img.max() - eo_img.min() + 1e-8)
        sar_img = x_np[3]                      # (H,W)

        axes[i,0].imshow(eo_img)
        axes[i,0].set_title("EO (pre-event)")
        axes[i,1].imshow(gt,   cmap="gray")
        axes[i,1].set_title("Ground Truth")
        axes[i,2].imshow(pred, cmap="gray")
        axes[i,2].set_title("Prediction")
        for ax in axes[i]:
            ax.axis("off")

    plt.suptitle(f"Sample Predictions — {split}", fontsize=14, y=1.01)
    plt.tight_layout()
    plt.savefig(f"{out_dir}/visualisations_{split}.png",
                dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Visualisations saved ✅")

    return {"iou": iou, "precision": prec, "recall": rec, "f1": f1}
