# PuzzArm_CRISPDM_Jt

This repository contains all required files for my PuzzArm machine learning project for ICTAII501 and ICTAII502.  
The goal of the project was to follow the CRISP-DM process and train a small image-classification model using the dataset supplied by the lecturer.

The focus of the assessment was on the workflow and documentation rather than building a perfect robot.  
The project includes data exploration, data preparation, modelling, evaluation, and supporting evidence.

---

## Repository Structure
PuzzArm_CRISPDM/
│
├── notebooks/
│ └── 
│
├── models/
│ └── puzzarm_model.pth
│
├── dataset_samples/
│ ├── 2/
│ ├── 4/
│ ├── 6/
│ └── Board/
│
├── evidence/
│ ├── training_plot.png
│ ├── dataset_overview.png
│ ├── repo_structure.png
│ └── model_results.png
│
├── template.md
└── README.md

---

## Project Summary

The PuzzArm project uses a small dataset of puzzle-piece images (“2”, “4”, “6”) and a set of board photos.  
A lightweight CNN was trained in Google Colab on this dataset to demonstrate the CRISP-DM process.

Because the dataset is very small, the model is not highly accurate — but it successfully demonstrates each step:  
Data Understanding > Preparation > Modelling > Evaluation > Deployment.

## Technologies Used

- Python 3  
- Google Colab  
- PyTorch  
- Torchvision  
- Matplotlib  

---

## Loading the Model

```python
import torch
from model import SimpleCNN

model = SimpleCNN(num_classes=4)
model.load_state_dict(torch.load("puzzarm_model.pth", map_location="cpu"))
model.eval()
