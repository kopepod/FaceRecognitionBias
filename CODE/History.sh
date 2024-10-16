python Main.py TrainModel --data_dir /mnt/TB2/PRAVSELS/ST1 --BS 128 --Device cuda:0


python Main.py TrainModel --data_dir /mnt/TB2/VGG-Face2/data/train --BS 128 Device cuda:3

python Main.py TrainModel --data_dir /mnt/TB2/VGG-Face2/data/train --BS 600 --Device cuda --imsize 256 --Split 0.9 --Epochs 10 --LR 0.00001

python Main.py TrainModel --data_dir /mnt/TB2/VGG-Face2/data/ --BS 1100 --Device cuda --imsize 256 --Split 0.9 --Epochs 5 --LR 1e-05 --ModelPath /mnt/GB480/facenet-pytorch-master_models/

python Main.py TrainModel --data_dir /mnt/TB2/VGG-Face2/data/ --BS 1100 --Device cuda --imsize 256 --Split 0.9 --Epochs 1 --LR 1e-05 --ModelPath /mnt/GB480/facenet-pytorch-master_models/

python Main.py TestModel --data_dir /mnt/TB2/VGG-Face2/data/ --BS 600 --Device cuda --imsize 256 --ModelPath /mnt/GB480/facenet-pytorch-master_models/FaceNet__2024_10_07_14_50_38.pth


python Main.py TrainModel --data_dir /mnt/TB2/SynthPar2_Runs/Run_0/ --BS 1100 --Device cuda --imsize 256 --Split 0.9 --Epochs 12 --LR 1e-05 --ModelPath /mnt/GB480/facenet-pytorch-master_models/


