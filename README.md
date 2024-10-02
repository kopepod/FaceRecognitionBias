# Face Recognition Bias
Alan Turing Research on Face Recognition Bias

<img src="https://github.com/kopepod/FaceRecognitionBias/blob/main/EXTRAS/FRbias.jpg" width="640" height="320" />

[![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1V2aAZe0Ljj3kjHAdcyXmBoPB9CUHWuYZ)
[![arXiv](https://img.shields.io/badge/arXiv-1234.56789-b31b1b.svg)]()
[![OneDrive](https://img.shields.io/badge/OneDrive-0078D4.svg?style=for-the-badge&logo=microsoftonedrive&logoColor=white)]()

This is the implementation of the paper entitled _Detecting Face Synthesis Using a Concealed Fusion Model_ this repository is divided as follows:

1. Create Environment
2. Dataset
3. Train Model
4. Compute Skin Scores

## Create Environment

To create the environment you have to install [anaconda](https://www.anaconda.com/download) and run the following command:
```bash
conda env create -f environment.yml
```
You should be able to run this command:
```bash
conda run -n FRBias python --version
```
## Datasets

You need two datasets to train the models.

### VGGFace2
Unfortunately this dataset is no longer avaiable from the original authors [VGGFace2_src](https://www.robots.ox.ac.uk/~vgg/data/vgg_face2/), we access the dataset via a torrent. It is necessary to download the [VGGFace2](https://academictorrents.com/details/535113b8395832f09121bc53ac85d7bc8ef6fa5b) dataset. To download the dataset run the following command.

```bash
transmission-cli "magnet:?xt=urn:btih:535113b8395832f09121bc53ac85d7bc8ef6fa5b&tr=https%3A%2F%2Facademictorrents.com%2Fannounce.php&tr=udp%3A%2F%2Ftracker.coppersurfer.tk%3A6969&tr=udp%3A%2F%2Ftracker.opentrackr.org%3A1337%2Fannounce"
```

### SynthPar2

The synthetic faces are avaibale here [HuggingFace](https://huggingface.co/datasets/pravsels/synthpar2/tree/main). You can download the dataset as:

```bash
bash DownloadSynthPar2.sh
```

### 


```bash
tree
```

## Train Models


We can pass specific parameters to train the models. Running the following command you will observe what the parameters control.

```bash
python Main.py TrainModel --help



```



