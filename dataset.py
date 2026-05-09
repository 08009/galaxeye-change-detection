
import os
import numpy as np
import torch
from torch.utils.data import Dataset
import rasterio
import albumentations as A
from albumentations.pytorch import ToTensorV2

def remap_labels(mask):
    """
    Original labels: 0=Background, 1=Intact, 2=Damaged, 3=Destroyed
    Remapped:        0,1 -> 0 (No-Change)   2,3 -> 1 (Change)
    Note: dataset already appears binary (0,1) but we apply
    remapping anyway so it is safe for any label set.
    """
    binary = np.zeros_like(mask, dtype=np.uint8)
    binary[mask == 2] = 1
    binary[mask == 3] = 1
    binary[mask == 1] = 0  # intact = no-change
    # if already binary (0=no-change, 1=change) this is still correct
    # because mask==1 gets 0 only if it meant "Intact"
    # BUT if mask is already binary 0/1 where 1=change, we must keep 1
    # So: safest approach — keep whatever is already 1 in a binary mask
    # We detect which case we are in:
    unique = np.unique(mask)
    if set(unique).issubset({0, 1}):
        # already binary — return as-is
        return mask.astype(np.uint8)
    else:
        # 4-class — remap
        out = np.zeros_like(mask, dtype=np.uint8)
        out[mask >= 2] = 1
        return out

def get_transforms(split, image_size):
    if split == "train":
        return A.Compose([
            A.Resize(image_size, image_size),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.RandomBrightnessContrast(p=0.3),
        ], additional_targets={"sar": "image"})
    else:
        return A.Compose([
            A.Resize(image_size, image_size),
        ], additional_targets={"sar": "image"})

class ChangeDetectionDataset(Dataset):
    def __init__(self, pre_dir, post_dir, target_dir, split, image_size=512):
        self.pre_dir    = pre_dir
        self.post_dir   = post_dir
        self.target_dir = target_dir
        self.split      = split
        self.image_size = image_size
        self.transform  = get_transforms(split, image_size)

        # Only use files that exist in ALL three folders
        # This handles the (1) duplicate issue in target
        pre_files    = set(os.listdir(pre_dir))
        post_files   = set(os.listdir(post_dir))
        target_files = set(os.listdir(target_dir))
        common = pre_files & post_files & target_files
        self.filenames = sorted(list(common))
        print(f"[{split}] Total usable samples: {len(self.filenames)}")

    def __len__(self):
        return len(self.filenames)

    def read_tif(self, path):
        with rasterio.open(path) as src:
            return src.read()  # (bands, H, W)

    def __getitem__(self, idx):
        fname = self.filenames[idx]

        # Read files
        eo  = self.read_tif(os.path.join(self.pre_dir,    fname))  # (3, H, W) uint8
        sar = self.read_tif(os.path.join(self.post_dir,   fname))  # (1, H, W) uint8
        tgt = self.read_tif(os.path.join(self.target_dir, fname))  # (1, H, W) uint8

        # Convert to HWC for albumentations
        eo_hwc  = eo.transpose(1, 2, 0).astype(np.float32)   # (H, W, 3)
        sar_hwc = sar.transpose(1, 2, 0).astype(np.float32)  # (H, W, 1)
        # Repeat SAR to 3 channels so albumentations treats it as image
        sar_hwc3 = np.repeat(sar_hwc, 3, axis=2)             # (H, W, 3)
        mask = tgt[0].astype(np.uint8)                        # (H, W)

        # Apply label remapping
        mask = remap_labels(mask)

        # Normalize EO: divide by 255
        eo_hwc = eo_hwc / 255.0

        # Normalize SAR: divide by 255 (uint8 0-238)
        sar_hwc3 = sar_hwc3 / 255.0

        # Apply augmentations
        augmented = self.transform(image=eo_hwc, sar=sar_hwc3, mask=mask)
        eo_aug  = augmented["image"]   # (H, W, 3)
        sar_aug = augmented["sar"]     # (H, W, 3)
        mask_aug = augmented["mask"]   # (H, W)

        # Convert to tensors: CHW
        eo_t  = torch.from_numpy(eo_aug.transpose(2, 0, 1)).float()   # (3, H, W)
        sar_t = torch.from_numpy(sar_aug[:, :, :1].transpose(2, 0, 1)).float()  # (1, H, W)
        mask_t = torch.from_numpy(mask_aug).float()                    # (H, W)

        # Concatenate EO + SAR -> 4-channel input
        x = torch.cat([eo_t, sar_t], dim=0)  # (4, H, W)

        return x, mask_t
