# Land & Water Segmentation — Technical Documentation

> **Project**: Binary semantic segmentation of land and water bodies from satellite/aerial imagery  
> **Environment**: Google Colab (Python 3, CUDA-enabled GPU — NVIDIA T4)  
> **Framework**: PyTorch + `segmentation-models-pytorch`  
> **Architecture**: U-Net with ResNet34 encoder (ImageNet pre-trained)

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Environment & Dependencies](#2-environment--dependencies)
3. [Dataset](#3-dataset)
4. [Data Preprocessing Pipeline](#4-data-preprocessing-pipeline)
5. [Dataset Splitting Strategy](#5-dataset-splitting-strategy)
6. [Class Imbalance Analysis](#6-class-imbalance-analysis)
7. [Mask Quality Control & Binarization](#7-mask-quality-control--binarization)
8. [Custom Dataset Class & DataLoaders](#8-custom-dataset-class--dataloaders)
9. [Model Architecture](#9-model-architecture)
10. [Loss Function](#10-loss-function)
11. [Optimizer & Learning Rate Scheduler](#11-optimizer--learning-rate-scheduler)
12. [Data Augmentation](#12-data-augmentation)
13. [Training Loop](#13-training-loop)
14. [Evaluation & Testing](#14-evaluation--testing)
15. [Inference on Unseen Images](#15-inference-on-unseen-images)
16. [File Naming & Path Conventions](#16-file-naming--path-conventions)
17. [Known Issues & Resolutions](#17-known-issues--resolutions)
18. [Full Pipeline Summary](#18-full-pipeline-summary)

---

## 1. Project Overview

This project trains a deep learning model to perform **binary semantic segmentation** on images containing water bodies (lakes, rivers, ponds, reservoirs, etc.). Each pixel in an image is classified as one of two classes:

| Class | Label Value | Mask Pixel Value |
|-------|-------------|-----------------|
| Land  | 0           | 0 (black)       |
| Water | 1           | 255 (white)     |

The model takes a 3-channel RGB image as input and outputs a single-channel binary mask. The final output is also used to compute **percentage coverage** of land vs. water within each image.

---

## 2. Environment & Dependencies

### Platform
- **Runtime**: Google Colab
- **Storage**: Google Drive (mounted at `/content/drive`)
- **Hardware**: NVIDIA T4 GPU (CUDA-enabled, verified in notebook)

### Python Libraries

| Library | Role |
|---------|------|
| `torch` | Core deep learning framework |
| `torchvision` | Image transforms, tensor utilities |
| `segmentation-models-pytorch` (`smp`) | Pre-built U-Net and encoder zoo |
| `albumentations` | Advanced data augmentation pipeline |
| `opencv-python` (`cv2`) | Image I/O, resizing, binarization |
| `numpy` | Numerical operations on arrays |
| `matplotlib` | Visualization of images and masks |
| `sklearn.model_selection` | `train_test_split` for dataset splitting |
| `tqdm` | Progress bars during training/testing |
| `PIL` (`Pillow`) | Image loading in Dataset class variants |
| `os`, `shutil` | File system and directory management |

### Installation Commands Used

```bash
pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cpu
pip install segmentation-models-pytorch
pip install opencv-python-headless
pip install albumentations
pip install matplotlib
pip install numpy
```

### GPU Check

```python
import torch
print("CUDA available:", torch.cuda.is_available())
print("Device name:", torch.cuda.get_device_name(0))
```

The notebook explicitly verifies CUDA availability before training.

---

## 3. Dataset

### Source

The dataset is stored in Google Drive at:

```
/content/drive/MyDrive/jatinrajani/Dataset/waterbodies/
```

### Raw Structure

```
waterbodies/
├── Images/          ← Raw RGB images (.jpg)
└── Mask/            ← Raw binary masks (.png)
```

### Image Characteristics

| Property | Value |
|----------|-------|
| Input format | JPEG (`.jpg`) |
| Color space | RGB |
| Raw resolution | Variable (resized to 256×256 during preprocessing) |
| Final resolution | 256×256 pixels |

### Mask Characteristics

| Property | Value |
|----------|-------|
| Mask format | PNG (`.png`) |
| Mode | Grayscale (single channel) |
| Raw pixel values | Variable (0–255) |
| Normalized values | Binary: 0 (land) or 255 (water) |
| Matching convention | Same stem name as image (e.g., `water_body_1234.png`) |

### Dataset Size

The total dataset comprises approximately **2,272 image–mask pairs** (inferred from the 80/10/10 split producing ~1,818 train / ~227 val / ~227 test samples as stated in the split verification cell).

### Naming Convention

Files follow the pattern:

```
water_body_<ID>.jpg   ← image
water_body_<ID>.png   ← mask (same numeric ID)
```

Example: `water_body_1475.jpg` ↔ `water_body_1475.png`

---

## 4. Data Preprocessing Pipeline

All preprocessing is performed in Python using OpenCV before training begins. The pipeline runs once and saves processed files to disk.

### Step 1 — Directory Setup

```python
os.makedirs(f'{base_dir}/train/images', exist_ok=True)
os.makedirs(f'{base_dir}/train/masks',  exist_ok=True)
os.makedirs(f'{base_dir}/val/images',   exist_ok=True)
os.makedirs(f'{base_dir}/val/masks',    exist_ok=True)
os.makedirs(f'{base_dir}/test/images',  exist_ok=True)
os.makedirs(f'{base_dir}/test/masks',   exist_ok=True)
```

A unified staging area `all/images` and `all/masks` is created first before the train/val/test split.

### Step 2 — Image Resizing

All images are resized to **256×256** pixels:

```python
img_resized  = cv2.resize(img,  (256, 256), interpolation=cv2.INTER_LINEAR)
mask_resized = cv2.resize(mask, (256, 256), interpolation=cv2.INTER_NEAREST)
```

- Images use **bilinear interpolation** (`INTER_LINEAR`) for smooth pixel blending.
- Masks use **nearest-neighbor interpolation** (`INTER_NEAREST`) to preserve hard binary boundaries.

### Step 3 — Mask Normalization

Raw masks may have arbitrary grayscale values. They are binarized to `{0, 1}` for training, then saved as `{0, 255}` for disk storage:

```python
mask_normalized = (mask_resized > 0).astype(np.uint8)   # {0, 1} in memory
cv2.imwrite(..., mask_normalized * 255)                  # {0, 255} on disk
```

### Step 4 — Strict Binarization Pass

A dedicated binarization sweep is run over all saved masks using OpenCV's threshold:

```python
_, mask_binarized = cv2.threshold(mask, 128, 255, cv2.THRESH_BINARY)
```

This catches any edge cases from JPEG artifacts or interpolation that might produce intermediate pixel values.

### Step 5 — Verification

After each processing step, the pipeline verifies:
- File counts match expected totals.
- Unique pixel values in masks are confirmed as `{0, 255}`.
- Sample image–mask pairs are visualized with `matplotlib`.

---

## 5. Dataset Splitting Strategy

### Primary Split (via `sklearn`)

An initial split uses `train_test_split` with `random_state=42`:

```python
train_files, temp_files = train_test_split(img_files, test_size=0.2, random_state=42)
val_files, test_files   = train_test_split(temp_files, test_size=0.5, random_state=42)
```

| Split      | Proportion | Approx. Count |
|------------|------------|---------------|
| Train      | 80%        | ~1,818        |
| Validation | 10%        | ~227          |
| Test       | 10%        | ~227          |

Files are physically **moved** (not copied) using `shutil.move` into their respective split folders.

### Re-split (Reproducible Random Shuffle)

Due to file naming mismatches discovered later (`.jpg` entries in text files pointing to `.png` masks), a new split is generated using Python's `random.shuffle` with `seed=42`:

```python
random.seed(42)
random.shuffle(all_files)

total = len(all_files)  # 2272
train_files = all_files[:int(0.8 * total)]           # 1818
val_files   = all_files[int(0.8 * total):int(0.9 * total)]  # 227
test_files  = all_files[int(0.9 * total):]            # 227
```

Results are saved to:
- `train_new.txt`
- `val_new.txt`
- `test_new.txt`

### Split File Format

Each `.txt` file contains one filename per line (e.g., `water_body_1234.png`), referencing the mask's PNG filename. The Dataset class internally resolves the corresponding `.jpg` image.

---

## 6. Class Imbalance Analysis

### Sampling

Class distribution is estimated from a sample of 20 training masks:

```python
mask = (mask > 0).astype(np.uint8)
water_percent = (np.sum(mask == 1) / mask.size) * 100
```

### Findings

| Class | Average Coverage | Class Weight |
|-------|-----------------|--------------|
| Land  | ~57.04%         | 1.7566       |
| Water | ~42.96%         | **2.3218**   |

The water class is the minority class. Weights are computed as the inverse of fractional coverage:

```
weight_water = 1 / (avg_water_percent / 100) = 2.3218
weight_land  = 1 / (avg_land_percent / 100)  = 1.7566
```

The water class weight (`2.3218`) is directly used in the loss function as `pos_weight`.

---

## 7. Mask Quality Control & Binarization

Due to file format mismatches and multi-step mask processing, a dedicated quality control workflow is maintained:

### Problem Identified

- Original split `.txt` files contained `.jpg` filenames, but masks were saved as `.png`.
- Some masks had intermediate grayscale values (not strictly 0 or 255) due to compression artifacts.

### Resolution

1. **Binarization loop** re-processes every mask file in all three splits.
2. **Validation loop** checks unique pixel values: any mask not in `{0, 255}` is flagged.
3. **New split files** (`train_new.txt`, `val_new.txt`, `test_new.txt`) use `.png` filenames exclusively.
4. The Dataset class maps filenames: `.png` → `.jpg` for images, keeps `.png` for masks.

### Canonical Mask Pipeline

```
Raw mask (variable values)
  ↓ cv2.threshold(mask, 128, 255, cv2.THRESH_BINARY)
Binarized mask {0, 255}
  ↓ save as .png
On-disk mask
  ↓ cv2.imread(..., cv2.IMREAD_GRAYSCALE)
  ↓ (mask > 0).astype(np.float32)
Training mask tensor {0.0, 1.0}
```

---

## 8. Custom Dataset Class & DataLoaders

### `WaterDataset` / `WaterBodyDataset` — Final Version

The final production Dataset class (used in the augmented training run) is:

```python
class WaterDataset(Dataset):
    def __init__(self, image_dir, mask_dir, file_list, transform=None):
        self.image_dir = image_dir
        self.mask_dir  = mask_dir
        self.file_list = file_list
        self.transform = transform

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        img_name  = self.file_list[idx]
        # Images are .jpg, masks are .png
        img_path  = os.path.join(self.image_dir, img_name)
        mask_name = img_name.rsplit('.', 1)[0] + '.png'
        mask_path = os.path.join(self.mask_dir, mask_name)

        img  = cv2.imread(img_path)
        img  = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask = (mask > 0).astype(np.float32)  # Binary float mask

        if self.transform:
            augmented = self.transform(image=img, mask=mask)
            img  = augmented['image']   # Tensor [C, H, W]
            mask = augmented['mask']    # Tensor [H, W]

        # Enforce correct channel dimensions
        if mask.dim() == 3 and mask.size(0) == 1:
            mask = mask.squeeze(0)
        elif mask.dim() == 2:
            mask = mask.unsqueeze(0)   # → [1, H, W]

        return img, mask
```

### Mask Channel Logic

| Condition | Action | Result |
|-----------|--------|--------|
| `mask.dim() == 3 and mask.size(0) == 1` | `squeeze(0)` | Remove spurious channel |
| `mask.dim() == 2` | `unsqueeze(0)` | Add required channel → `[1, H, W]` |

This ensures the mask tensor always has shape `[1, H, W]` to match model output.

### None Filtering

Items that fail to load (corrupted files, missing pairs) return `None` and are removed via a post-initialization filter:

```python
def filter_none(dataset):
    filtered = [i for i in range(len(dataset)) if dataset[i] is not None]
    return torch.utils.data.Subset(dataset, filtered)
```

### DataLoader Configuration

| Parameter     | Value |
|---------------|-------|
| `batch_size`  | 16    |
| `shuffle`     | `True` (train only) |
| `num_workers` | 2     |

```python
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True,  num_workers=2)
val_loader   = DataLoader(val_dataset,   batch_size=16, shuffle=False, num_workers=2)
test_loader  = DataLoader(test_dataset,  batch_size=16, shuffle=False, num_workers=2)
```

### Normalization (ImageNet Statistics)

Applied via `albumentations.Normalize` in the final training run:

```python
mean = [0.485, 0.456, 0.406]
std  = [0.229, 0.224, 0.225]
```

These are the standard ImageNet mean and standard deviation values, appropriate since the encoder uses ImageNet pre-trained weights.

---

## 9. Model Architecture

### Framework

`segmentation-models-pytorch` (`smp`) v≥0.3

### Architecture: U-Net

U-Net is an encoder–decoder architecture with skip connections, originally developed for biomedical image segmentation. It excels at preserving fine spatial detail while capturing high-level semantic context.

```
Input [B, 3, H, W]
       ↓
  ┌─── Encoder (ResNet34) ─────────────────────────────────┐
  │  Block 1: Conv 64  → stride 1  (H×W)                   │
  │  Block 2: Conv 64  → stride 2  (H/2 × W/2)             │
  │  Block 3: Conv 128 → stride 2  (H/4 × W/4)             │
  │  Block 4: Conv 256 → stride 2  (H/8 × W/8)             │
  │  Block 5: Conv 512 → stride 2  (H/16 × W/16)           │
  └────────────────────────────────────────────────────────┘
       ↓ (bottleneck)
  ┌─── Decoder (SMP default) ──────────────────────────────┐
  │  Upsample + skip from Block 5 → 256 channels           │
  │  Upsample + skip from Block 4 → 128 channels           │
  │  Upsample + skip from Block 3 → 64 channels            │
  │  Upsample + skip from Block 2 → 32 channels            │
  │  Upsample + skip from Block 1 → 16 channels            │
  └────────────────────────────────────────────────────────┘
       ↓
  Segmentation Head: Conv 1×1 → 1 channel (logit)
       ↓
Output [B, 1, H, W]  ← raw logit (no activation)
```

### Model Initialization

```python
model = smp.Unet(
    encoder_name    = 'resnet34',    # Encoder backbone
    encoder_weights = 'imagenet',    # Transfer learning weights
    in_channels     = 3,             # RGB input
    classes         = 1,             # Single binary output channel
    activation      = None           # Logits output (for BCEWithLogitsLoss)
)
```

### Encoder: ResNet34

| Property | Value |
|----------|-------|
| Architecture | ResNet-34 (34 layers) |
| Pre-training | ImageNet (1000-class classification) |
| Trainable | Yes (entire model fine-tuned end-to-end) |
| Output stages | 5 feature maps at different scales |

ResNet34 uses residual (skip) connections in the encoder to allow gradients to flow through many layers without vanishing:

```
x → [Conv → BN → ReLU → Conv → BN] + x → ReLU
```

### Parameter Count

The full model contains approximately **21 million trainable parameters** (ResNet34 encoder: ~21M, decoder: ~4M).

---

## 10. Loss Function

### Weighted Binary Cross-Entropy with Logits

```python
class_weights = torch.tensor([2.3218]).to(device)
criterion = nn.BCEWithLogitsLoss(pos_weight=class_weights)
```

`BCEWithLogitsLoss` combines a **sigmoid activation** and **binary cross-entropy** in a single numerically stable operation:

```
loss = -[pos_weight × y × log(σ(x)) + (1 − y) × log(1 − σ(x))]
```

Where:
- `x` = raw model logit output
- `y` = ground truth label (0 or 1)
- `pos_weight = 2.3218` = weight applied to positive (water) class predictions

### Why `BCEWithLogitsLoss`

- Numerically more stable than applying `sigmoid` then `BCELoss` separately.
- `pos_weight` directly addresses the class imbalance identified in Section 6.
- `activation=None` in the model declaration is required; adding sigmoid in the model would cause double sigmoid during loss computation.

---

## 11. Optimizer & Learning Rate Scheduler

### Optimizer: Adam

```python
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
```

| Parameter | Value |
|-----------|-------|
| Algorithm | Adam (Adaptive Moment Estimation) |
| Learning rate | `1e-3` (0.001) |
| β₁ (momentum) | 0.9 (default) |
| β₂ (RMS term) | 0.999 (default) |
| ε (stability) | 1e-8 (default) |
| Weight decay | 0 (default) |

### Scheduler: ReduceLROnPlateau

```python
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.1, patience=5
)
```

| Parameter | Value | Meaning |
|-----------|-------|---------|
| `mode`    | `'min'` | Monitor for decreasing metric |
| `factor`  | `0.1` | Multiply LR by 0.1 on plateau |
| `patience`| `5` | Wait 5 epochs before reducing |
| Monitored metric | Validation loss | Reduced when `val_loss` stops improving |

The scheduler is stepped at the end of each epoch:

```python
scheduler.step(val_loss)
```

This means after 5 consecutive epochs without validation loss improvement, the learning rate drops from `1e-3` → `1e-4`, and can drop again to `1e-5` if stagnation continues.

---

## 12. Data Augmentation

### Library: Albumentations

```python
import albumentations as A
from albumentations.pytorch import ToTensorV2
```

### Final Augmentation Pipeline

```python
transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.Rotate(limit=10, p=0.5),
    A.ColorJitter(brightness=0.2, contrast=0.2,
                  saturation=0.2, hue=0.2, p=0.5),
    A.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]),
    ToTensorV2(),
], additional_targets={'mask': 'mask'})
```

### Augmentation Details

| Transform | Parameters | Purpose |
|-----------|-----------|---------|
| `HorizontalFlip` | p=0.5 | Mirror left–right (water bodies are symmetric) |
| `VerticalFlip` | p=0.5 | Mirror top–bottom |
| `Rotate` | limit=10°, p=0.5 | Small angle rotation for orientation variance |
| `ColorJitter` | b=0.2, c=0.2, s=0.2, h=0.2, p=0.5 | Handle lighting/sensor variation |
| `Normalize` | ImageNet stats | Align with encoder pre-training distribution |
| `ToTensorV2` | — | Convert NumPy HWC → PyTorch CHW tensor |

### `additional_targets` Usage

```python
A.Compose([...], additional_targets={'mask': 'mask'})
```

This ensures spatial transformations (flip, rotate) are **applied identically** to both the image and its mask, maintaining pixel-wise correspondence.

### Early Training (No Augmentation)

The initial training runs used only basic `transforms.ToTensor()` without normalization or augmentation. The augmented pipeline was introduced in the second (refined) training phase.

---

## 13. Training Loop

### Configuration

| Hyperparameter | Value |
|---------------|-------|
| Epochs | 20 |
| Batch size | 16 |
| Optimizer | Adam |
| Initial LR | 1e-3 |
| Loss | BCEWithLogitsLoss (pos_weight=2.3218) |
| Early stopping | No (but best model saved by val_loss) |

### Training Phase (per epoch)

```python
model.train()
for images, masks in train_loader:
    images, masks = images.to(device), masks.to(device)
    optimizer.zero_grad()
    outputs = model(images)                  # Forward pass
    loss = criterion(outputs, masks)         # Weighted BCE loss
    loss.backward()                          # Backpropagation
    optimizer.step()                         # Weight update
    preds = torch.sigmoid(outputs) > 0.5    # Threshold predictions
```

### Validation Phase (per epoch)

```python
model.eval()
with torch.no_grad():
    for images, masks in val_loader:
        outputs = model(images)
        loss = criterion(outputs, masks)
        preds = torch.sigmoid(outputs) > 0.5
```

### Metrics Computed

| Metric | Formula | Tracked |
|--------|---------|---------|
| Train Loss | BCE loss averaged over training set | Per epoch |
| Train Accuracy | Pixel-level correct / total pixels | Per epoch |
| Val Loss | BCE loss averaged over validation set | Per epoch |
| Val Accuracy | Pixel-level correct / total pixels | Per epoch |

**Pixel-level accuracy** is computed as:
```python
train_acc = train_correct / train_total
# where: train_correct = (preds == masks).float().sum()
#        train_total   = masks.numel()  ← total pixels across batch
```

### Model Checkpointing

The best model (lowest validation loss) is saved to disk after each improvement:

```python
best_model_path = '/content/drive/MyDrive/jatinrajani/Dataset/best_model.pth'
# (augmented run saves to:)
new_best_model_path = '.../best_model_augmented.pth'

if val_loss < best_val_loss:
    best_val_loss = val_loss
    torch.save(model.state_dict(), best_model_path)
```

Two model checkpoints exist:
- `best_model.pth` — Saved from initial (non-augmented) training run
- `best_model_augmented.pth` — Saved from augmented training run (final model)

### Training on Augmented Data — Pre-loading Best Model

The augmented training loop **loads the previous best model** before starting, enabling fine-tuning rather than training from scratch:

```python
model.load_state_dict(torch.load(best_model_path))
```

---

## 14. Evaluation & Testing

### Test Set Loading

```python
test_dataset = WaterDataset(
    image_dir = f'{base_dir}/test/images',
    mask_dir  = f'{base_dir}/test/masks',
    file_list = test_new_files,
    transform = transform
)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
```

### Test Accuracy Computation

```python
model.load_state_dict(torch.load(best_model_path))
model.eval()

test_correct = 0
test_total   = 0
with torch.no_grad():
    for images, masks in test_loader:
        images, masks = images.to(device), masks.to(device)
        outputs = model(images)
        preds   = torch.sigmoid(outputs) > 0.5
        test_correct += (preds == masks).float().sum()
        test_total   += masks.numel()

test_acc = test_correct / test_total
```

### Average Land/Water Percentage Over Test Set

A separate evaluation pass computes the average predicted coverage per image:

```python
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
for images, _ in test_loader:
    outputs   = model(images.to(device))
    preds     = torch.sigmoid(outputs) > 0.5
    pred_mask = preds[0].cpu().numpy().flatten()
    water_pixels = pred_mask.sum()
    water_percent = (water_pixels / pred_mask.size) * 100
```

This gives a macro-average of predicted water and land percentages across the entire test set.

---

## 15. Inference on Unseen Images

### Input Handling

An unseen image can be uploaded via Colab's `files.upload()`:

```python
from google.colab import files
uploaded = files.upload()
image_file = list(uploaded.keys())[0]
```

### Preprocessing for Inference

Images are preprocessed to **224×224** in the final inference cell (note: training used 256×256; the 224×224 resize is used in the last inference-only cell):

```python
img = cv2.imread(image_file)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_LINEAR)

transform = A.Compose([
    A.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]),
    ToTensorV2(),
])

input_tensor = transform(image=img)['image'].unsqueeze(0).to(device)
```

> **Note**: The training preprocessing used 256×256; the inference cell uses 224×224. Ensure consistency between training and inference resolution in production.

### Prediction & Thresholding

```python
with torch.no_grad():
    output    = model(input_tensor)
    pred_mask = torch.sigmoid(output) > 0.5
```

- Raw output: logit (unbounded float)
- `sigmoid(output)`: probability ∈ [0, 1] per pixel
- `> 0.5`: binary mask (threshold)

### Coverage Calculation

```python
pred_mask_np   = pred_mask.squeeze().cpu().numpy().astype(np.uint8)
total_pixels   = 224 * 224
water_pixels   = np.sum(pred_mask_np)
land_pixels    = total_pixels - water_pixels
water_percent  = (water_pixels / total_pixels) * 100
land_percent   = (land_pixels  / total_pixels) * 100
```

### Visualization

Three panels are displayed side-by-side:

1. **Original Image** — RGB input resized to inference resolution
2. **Predicted Mask** — Grayscale binary (land=black, water=white)
3. **Overlay** — Original image with water pixels colored red (`[0, 0, 255]` in BGR)

```python
overlay = img_resized.copy()
overlay[pred_mask == 1] = [0, 0, 255]   # Water pixels → red
```

---

## 16. File Naming & Path Conventions

### Directory Tree (Final)

```
/content/drive/MyDrive/jatinrajani/Dataset/
├── waterbodies/
│   ├── Images/                ← Original raw images (.jpg)
│   ├── Mask/                  ← Original raw masks (.png / .jpg)
│   ├── all/
│   │   ├── images/            ← Staging area after preprocessing
│   │   └── masks/
│   ├── train/
│   │   ├── images/            ← Final training images
│   │   └── masks/             ← Final training masks
│   ├── val/
│   │   ├── images/
│   │   └── masks/
│   ├── test/
│   │   ├── images/
│   │   └── masks/
│   ├── train.txt              ← Original split list
│   ├── val.txt
│   ├── test.txt
│   ├── train_new.txt          ← Corrected split list (PNG filenames)
│   ├── val_new.txt
│   └── test_new.txt
└── best_model.pth             ← Non-augmented training checkpoint
    best_model_augmented.pth   ← Augmented training checkpoint (FINAL)
```

### Extension Convention

| File Type | Extension | Notes |
|-----------|-----------|-------|
| Images | `.jpg` | JPEG, RGB, 256×256 |
| Masks | `.png` | PNG, Grayscale, 256×256, binary {0,255} |
| Split files | `.txt` | One filename per line, PNG extension |
| Models | `.pth` | PyTorch `state_dict` format |

---

## 17. Known Issues & Resolutions

| Issue | Root Cause | Resolution |
|-------|-----------|------------|
| Masks not loading | `.jpg` filenames in split `.txt` but masks saved as `.png` | Dataset `__getitem__` maps: `img_name.rsplit('.', 1)[0] + '.png'` |
| Non-binary mask values | Bilinear resize on masks or JPEG compression artifacts | Explicit `cv2.threshold(mask, 128, 255, cv2.THRESH_BINARY)` sweep |
| Mask size mismatch | Original masks not resized | Dedicated resize pass to enforce 256×256 on all masks |
| `val.txt`/`test.txt` mismatch | Generated with `.jpg` names, masks stored as `.png` | Rebuilt as `val_new.txt`, `test_new.txt` with PNG filenames |
| 224 vs 256 resolution inconsistency | Inference cell used 224×224 while training used 256×256 | Training resolution is 256×256; inference cell should match |
| `verbose=True` deprecated in scheduler | PyTorch version change | Removed `verbose=True` from `ReduceLROnPlateau` |
| `None` samples from DataLoader | Missing/corrupted image–mask pairs | `filter_none()` post-initialization filtering |
| Model loading on CPU Colab instance | CUDA tensors saved, CPU instance loaded | `torch.load(..., map_location='cpu')` |

---

## 18. Full Pipeline Summary

```
1. Mount Google Drive
          ↓
2. Explore raw dataset (Images/ + Mask/)
          ↓
3. Resize images & masks to 256×256
   Binarize masks → {0, 255}
   Save to all/images, all/masks
          ↓
4. Train/Val/Test split (80/10/10, seed=42)
   Move files to train/, val/, test/ folders
   Save split lists → train.txt, val.txt, test.txt
          ↓
5. Analyze class imbalance
   → Water weight: 2.3218
          ↓
6. Re-binarization QC sweep (threshold=128)
   Rebuild split lists with .png extension
   → train_new.txt, val_new.txt, test_new.txt
          ↓
7. Build WaterDataset + DataLoaders
   batch_size=16, num_workers=2
          ↓
8. Initialize U-Net (ResNet34, ImageNet weights)
   ~21M trainable parameters
          ↓
9. Define loss: BCEWithLogitsLoss(pos_weight=2.3218)
   Optimizer: Adam(lr=1e-3)
   Scheduler: ReduceLROnPlateau(factor=0.1, patience=5)
          ↓
10. Train 20 epochs (no augmentation)
    Track train/val loss & pixel accuracy
    Save best_model.pth
          ↓
11. Add Albumentations augmentation pipeline
    Load best_model.pth → fine-tune for 20 more epochs
    Save best_model_augmented.pth
          ↓
12. Evaluate on test set
    Metric: pixel-level accuracy
    Compute average land/water % predictions
          ↓
13. Inference on unseen image
    Upload → Preprocess → Predict → Visualize
    Output: land %, water %, overlay visualization
```

---

*Documentation generated from `land_water_segmentation.ipynb`. All code references are sourced directly from the notebook cells.*
