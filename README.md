# FR_BIAS
Alan Turing Research on Face Recognition Bias

<img src="https://github.com/kopepod/FaceRecognitionBias/blob/main/EXTRAS/Issue0.jpg" width="420" height="420" />
<img src="https://github.com/kopepod/FaceRecognitionBias/blob/main/EXTRAS/Issue1.jpg" width="420" height="420" />

[![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1elztvJ-Wz1qsFMH2T-uc1y-V-7-3An7i)
[![arXiv](https://img.shields.io/badge/arXiv-1234.56789-b31b1b.svg)](https://arxiv.org/abs/2401.04257)
[![OneDrive](https://img.shields.io/badge/OneDrive-0078D4.svg?style=for-the-badge&logo=microsoftonedrive&logoColor=white)](https://livewarwickac-my.sharepoint.com/:f:/g/personal/u1873231_live_warwick_ac_uk/Eghl0WXMTJJJmHu4KEWIL8QBP0yyraEjpd6FYSOpqLMzAw?e=w7cvtw)

This is the implementation of the paper entitled _Detecting Face Synthesis Using a Concealed Fusion Model_ this repository is divided as follows:

1. Create Environment
2. Dataset
3. Train Models Bank
4. Train Fusion Model
5. Test Models

## Create Environment

To create the environment you have to install [anaconda](https://www.anaconda.com/download) and run the following command:
```bash
conda env create -f environment.yml
```
You should be able to run this command:
```bash
conda run -n bayesianfusion python --version
```
## Dataset

It is necessary to download the [FFHQ](https://github.com/NVlabs/ffhq-dataset) dataset from the original authors.  We also provide the synthetic images we generated on  [OneDrive](https://livewarwickac-my.sharepoint.com/:f:/g/personal/u1873231_live_warwick_ac_uk/Eghl0WXMTJJJmHu4KEWIL8QBP0yyraEjpd6FYSOpqLMzAw?e=w7cvtw). Create the following structure with the Real and Synthetic Images:

```bash
tree
.
└── FFHQ
    ├── Fake
    │       ├── seed0000.png
    │       ├── seed0001.png
    │       └── seed0002.png
    └── Real
            ├── 00000.png
            ├── 00001.png
            └── 00002.png
```
Where the Real is the folder comprises the FFHQ samples and the Fake the ones generated from the synthesizer, respectively.

## Train Models

The model bank requires training separately each feature model and then produce the scores individually.

We can pass specific parameters to train the models. Running the following command you will observe what the parameters control.

```bash
python Main.py TrainBankModel --help

usage: Fake Synthesis Detection via Late Fusion TrainBankModel [-h] [--DatasetPath DATASETPATH] [--Model MODEL] [--FileName FILENAME] [--Criterion CRITERION] [--Optimizer OPTIMIZER] [--Scheduler SCHEDULER] [--Epochs EPOCHS] [--LR LR]
                                                               [--BS BS] [--Device DEVICE] [--Weights WEIGHTS] [--AccStop ACCSTOP] [--Split SPLIT]

Fake Synthesis Detection: this is the code to train several models to detect face synthesis.

options:
  -h, --help            show this help message and exit
  --DatasetPath DATASETPATH
                        Dataset path
  --Model MODEL         {RS-18, RS-34, MLP_cpcf3, MLP_c3n1f3, VGG19, SNet, DenseNet101, RSX-101, VT16}
  --FileName FILENAME   Model 2 Save - default ModelName_Time
  --Criterion CRITERION
                        {CrossEntropyLoss, MSELoss}
  --Optimizer OPTIMIZER
                        {SGD, Adam}
  --Scheduler SCHEDULER
                        {StepLR, MultiStepLR}
  --Epochs EPOCHS       Number of Epochs = 3
  --LR LR               Learing Rate = 0.001
  --BS BS               Batch Size = 256
  --Device DEVICE       GPU device
  --Weights WEIGHTS     {None, 'DEFAULT', 'IMAGENET1K_V1', 'IMAGENET1K_V2'}
  --AccStop ACCSTOP     Accuracy Stop = 0.95
  --Split SPLIT         Split size = 0.8

```

This will save the model on the FileName path with the prefix name for training, e.g., RS-34. Place all models in the same path to fuse features.

For example to train a model from the bank use:

```bash
python Main.py TrainBankModel --DatasetPath <>  --Model RS-34 --Device "cuda:0" --LR 0.01 --BS 512
```

To generate the features from each individual model use:

```bash
python Main.py FuseFeatures --DatasetPath <> --ModelsFilePath ./MODELS/
```





