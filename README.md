# EE641 — Homework 2

## Name: Aditi Surendra Tarate
## USC ID: 4251439505
## USC Email: tarate@usc.edu

This repository contains two problems:

Problem 1 – Multi-Scale Single-Shot Detector

Problem 2 – Keypoint Detection: Heatmap vs Direct Regression

Repository Layout:

```text
ee641-hw1-aditi-2400/
├─ report.pdf
├─ problem1/
│  ├─ train.py
│  ├─ evaluate.py
│  ├─ model.py
│  ├─ loss.py
│  ├─ utils.py
│  ├─ dataset.py
│  └─ results/                # created automatically
│
├─ problem2/
│  ├─ train.py
│  ├─ evaluate.py
│  ├─ model.py
│  ├─ baseline.py
│  ├─ dataset.py
│  └─ results/                # created automatically
│
└─ datasets/
   ├─ detection/
   │  ├─ train/              
   │  ├─ val/                 
   │  ├─ train_annotations.json
   │  └─ val_annotations.json
   │
   └─ keypoints/
      ├─ train/               
      ├─ val/                
      ├─ train_annotations.json
      └─ val_annotations.json
