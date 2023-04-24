Dilated Involutional Pyramid Network (DInPNet): A Novel Model for Printed Circuit Board (PCB) Components Classification
==============================

## Overview:

This repository contains the source code of our paper, DInPNet (published in <a href="https://www.isqed.org/">ISQED-23</a>).

We introduce a novel light-weight PCB component classification network, named DInPNet. We introduce the dilated involutional pyramid (DInP) block, which consists of an involution for transforming the input feature map into a low-dimensional space for reduced computational cost, followed by a pairwise pyramidal fusion of dilated involutions that resample back the feature map. This enables learning representations for a large effective receptive field while bringing down the number of parameters considerably.

<hr>

Project Organization
------------
    ├── LICENSE                         <- The LICENSE for developers using this project.
    ├── README.md                       <- The top-level README for developers using this project.
    ├── requirements.txt                <- The requirements file for reproducing the analysis environment, e.g. generated with `pip freeze > requirements.txt`.
    |── reports                         <- The directory containing metadata used for repo.
    ├── checkpoints                     <- Directory where best models will be saved.
    ├── src                             <- Source code for use in this project.
    │   ├── dataloader.py               <- Source code for generating data loader.
    |   ├── config.py                   <- basic configurations for classification training of DInPNet model.
    │   ├── network.py                  <- Source code for the DInPNet network.
    │   ├── utils.py                    <- Source code for utilities and helper functions.
    │   ├── train.py                    <- Source code for training and validation of DInPNet
    └─────────────────────────────────────────────────────────────────────────────────────────────────────────────
------------

<hr>

## Network Architecture

<img src="reports/figures/Arch_Diag.png" width=100%>

<p align="center"> Figure 1. (A) DInPNet (B) Dilated Involutional Pyramid Block </p>

<hr>

## Get Started

Dependencies:

```bash
pip install -r requirements.txt
```
### (Optional) Conda Environment Configuration

First, create a conda environment
```bash
conda create -n va python=3.8
conda activate va
conda install pip
pip install -r requirements.txt
```

### Dataset

We have used FICS-PCB dataset which can be downloaded from <a href="https://www.trust-hub.org/#/data/fics-pcb">here</a>. components data needs to placed under `data/` directory.

Data Structure in `data/` directory after completing above steps
------------
    ├── Train
    │   ├───capacitors
    │   │   └── image_0.png
    │   │   └── image_1.png
    │   │   └── ...
    │   ├───diodes
    │   │   └── image_0.png
    │   │   └── image_1.png
    │   │   └── ...
    |   └── ...
    ├── Test
    │   ├───capacitors
    │   │   └── image_0.png
    │   │   └── image_1.png
    │   │   └── ...
    │   ├───diodes
    │   │   └── image_0.png
    │   │   └── image_1.png
    │   │   └── ...
    |   └── ...
    └─────────────────────────────────────────────────────────────────────────────────────────────────────────────
------------

### Train model

change the hyperparameters and configuration parameters according to need in ```src/config.py```.

To train DInPNet, Run following command from ```/src``` directory.

```bash
python train.py
``` 
Above command will train model for 100 epochs with given configuration.

The trained checkpoint for model training will be saved in ```/weights/best.pt```

## Citation
```
@inproceedings {mantravadi2023Dilated,
    title            = {{Dilated Involutional Pyramid Network (DInPNet): A Novel Model for Printed Circuit Board (PCB) Components Classification}},
	year             = "2023",
	author           = "Ananya Mantravadi and Dhruv Makwana and R Sai Chandra Teja and Sparsh Mittal and Rekha Singhal",
	booktitle        = {{24th International Symposium on Quality Electronic Design (ISQED)}},
	address          = "California, USA",
}
```
## License
<hr>
CC BY-NC-ND 4.0