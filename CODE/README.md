# Face Recognition Bias

This Section is organized as follows:

1. Data Preparation
2. 


## Train Models

We can pass specific parameters to train the models. Running the following command you will observe what the parameters control.

```bash
python Main.py TrainModel --help
```

To train the model with the provided data run as:

```bash
python Main.py TrainModel --data_dir ./SAMPLES/VGGFace2_Small --BS 1100 --Device cuda --imsize 256 --Split 0.9 --Epochs 12 --LR 1e-05 --ModelPath ./MODELS/
```


