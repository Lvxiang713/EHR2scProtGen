# EHR2scProtGen

A PyTorch-based pipeline for generating single-cell protein expression profiles conditioned on electronic health record (EHR) embeddings, combining a CLIP‐style contrastive learning with a Gaussian diffusion model.

---
<img width="7874" height="10236" alt="Figure1拼接" src="https://github.com/user-attachments/assets/381b0974-d927-4f90-98bf-7387188bf4fb" />


## Table of Contents

- [Overview](#overview)  
- [Repository Structure](#repository-structure)  
- [Usage](#usage)    
- [Example Notebook](#example-notebook)  
- [Acknowledgements](#acknowledgements)  

---

## Overview

This project implements:

1. Contrastive learning model structure  
   Learns joint embeddings of EHR vectors and single-cell protein profiles.  

2. Diffusion model  
   A scAtt-Net operating on per‐cell protein features, with timestep embeddings and cross‐attention conditioning on EHR..
   
   ---

## Repository Structure\
```
├── checkpoints/ # Saved model checkpoints
├── configs/
│ ├── contrastive.yaml # Contrastive  learning model settings
│ └── diffusion.yaml # Diffusion model settings
├── src/
│ ├── contrastive_src/ # CLIP encoder implementation
│ │ ├── configs/
│ │ ├── datasets/
│ │ └── models/
│ └── diffusion_src/ # Diffusion model
│ ├── models/
│ ├── utils/
├── notebook/ #The downstream analysis
├── data/ # sampled datasets and trained EHR embbeding
├── example/ # Jupyter notebooks demonstrating generating sample
├── requirements.txt # enviroment package
├── README.md
└── generate.py # generation script
```
---

## Usage
1. Create a new Conda environment 
   ```
   conda create -n ehr2scprotgen python=3.9.18
   conda activate ehr2scprotgen
   pip install -r requirements.txt
   ```
 2. Run the generation script
   ```
   python generate.py
   ```
   Then you will get the synthesized data in the sampled_cells folder.

---
## Example Notebook

A demo notebook is available under example/GenerateExample.ipynb showing:
The procession of loading configs, generating cells conditioned on trained EHR embedding.

## Downstream Analysis Notebook

The contents of notebook folder demonstrates how to evaluate and analyze the generated single-cell data downstream, 
including DataDistribution.ipynb and DownstreamAnalysis.ipynb

## Acknowledgements
Data used in this project can be acquired from the acquired from the database (https://hpap.pmacs.upenn.edu/) of the Human Pancreas Analysis Program (HPAP; RRID:SCR_016202; PMID: 31127054; PMID: 36206763). HPAP is part of a Human Islet Research Network (RRID:SCR_014393) consortium (UC4-DK112217, U01-DK123594, UC4-DK112232, and U01-DK123716).



