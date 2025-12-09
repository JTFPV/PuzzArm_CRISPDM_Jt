# PuzzArm_CRISPDM_Jt

This repo has my PuzzArm machine learning project for **ICTAII501** and **ICTAII502**.  
The idea was to follow the **CRISP-DM** steps and train a small image classifier on a tiny
dataset of digit-shaped puzzle pieces (0–9). The point is the process, not building a
perfect model.

---

## What’s here

- **notebooks/** – data understanding + training / testing notebooks  
- **models/**  
  - `puzzarm_cnn_model.pth` – simple CNN baseline  
  - `puzzarm_mobilenet_v2.pth` – MobileNetV2 (final model)  
- **dataset_samples/** – example digit images  
- **evidence/** – plots, confusion matrices, screenshots of the demo  
- **template.md** – CRISP-DM write-up  
- **README.md** – this file

---

## Models & results

Final processed dataset = **50 images** (5 per digit 0–9), split roughly 70/20/10 into
train / validation / test.

I trained two different models:

1. **SimpleCNN (baseline)**  
   - Small custom CNN built from scratch.  
   - **Test accuracy:** `0.0` (0/5 correct).  
   - With only 5 test images the accuracy moves in 20% jumps, and this one basically
     failed to generalise.

2. **MobileNetV2 (transfer learning)**  
   - Pretrained MobileNetV2 from Torchvision, final layer swapped to 10 classes.  
   - **Test accuracy:** `0.4` (2/5 correct).  
   - Confusion matrices and training curves are saved in `evidence/`.

MobileNetV2 clearly does better than the simple CNN, so that’s the model I treat as the
“final” one for this project.

---

## Deployment / demo

The main training notebook also has a small **inference demo**:

- Rebuilds MobileNetV2 and loads `puzzarm_mobilenet_v2.pth`.  
- Uses the same preprocessing as training (grayscale → 128×128 → normalise).  
- Reads a folder of **new digit photos** that weren’t in the train/test split.  
- Shows each image with the **predicted digit + class probabilities**.

Screenshots of this are in `evidence/` as proof that the model actually runs on new
images, not just the training set.

---

## Quick example: loading the final model

```python
import torch
from torchvision import models
import torch.nn as nn

num_classes = 10

model = models.mobilenet_v2(weights=None)
in_features = model.classifier[1].in_features
model.classifier[1] = nn.Linear(in_features, num_classes)

state_dict = torch.load("models/puzzarm_mobilenet_v2.pth", map_location="cpu")
model.load_state_dict(state_dict)
model.eval()
