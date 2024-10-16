# Face Recognition Bias

This Section is organized as follows:

1. Data Preparation
2. Train Models
3. Resume Training


## 1. Data Preparation

The training dataset must have a class label per folder and then the samples per class. Here is the one provided.

```bash 
tree SAMPLES/

SAMPLES/
└── VGGFace2_Small
    ├── n000001
    │   ├── 0001_01.jpg
    │   ├── 0002_01.jpg
    │   ├── 0003_01.jpg
... more
    │   ├── 0009_01.jpg
    │   └── 0010_01.jpg
    ├── n000009
    │   ├── 0001_01.jpg
    │   ├── 0003_01.jpg
... more
    │   ├── 0009_01.jpg
    │   └── 0010_01.jpg
    ├── n000029
    ├── n000040
    ├── n000078
    ├── n000082
... more
    ├── n000148
    └── n000149

```

## 2. Train Models

We can pass specific parameters to train the models. By running the following command you will observe what the parameters control the training process.

```bash
python Main.py TrainModel --help

[start] 2024-10-16 11:50:02.694943

usage: FaceNet Bias Library TrainModel [-h] --data_dir DATA_DIR [--BS BS] [--LR LR] [--Split SPLIT]
                                       [--imsize IMSIZE] [--Epochs EPOCHS] [--Device DEVICE] [--Workers WORKERS]
                                       [--ModelPath MODELPATH] [--Resume] [--TFWritter] [--Visualize]

Train FaceNet model

options:
  -h, --help            show this help message and exit
  --data_dir DATA_DIR   Dataset location
  --BS BS               Batch size
  --LR LR               Learning Rate
  --Split SPLIT         Train/Validate split
  --imsize IMSIZE       Image resize samples
  --Epochs EPOCHS       Number of epochs
  --Device DEVICE       Device
  --Workers WORKERS     Number of CPU workers
  --ModelPath MODELPATH
                        Path to save trained models
  --Resume              Verbose (False) resume and overwrite model
  --TFWritter           Tensor Flow log writer
  --Visualize           Visualize (False)

```

To train the model with the provided data run as:

```bash
python Main.py TrainModel --data_dir ./SAMPLES/VGGFace2_Small --BS 1100 --Device cuda --imsize 256 --Split 0.9 --Epochs 12 --LR 1e-05 --ModelPath ./TrainedMODELS/
```

## 3. Resume Training
