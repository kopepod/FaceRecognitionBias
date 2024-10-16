echo "Full Fine-Tune on SynthPar2\n\n"

python Main.py TrainModel --data_dir /mnt/TB2/SynthPar2_Runs/Run_1/ --BS 1100 --Device cuda --imsize 256 --Split 0.9 --Epochs 7 --LR 1e-05 --ModelPath /mnt/GB480/facenet-pytorch-master_models/FaceNet__2024_10_08_15_13_36.pt --Resume

python Main.py TrainModel --data_dir /mnt/TB2/SynthPar2_Runs/Run_2/ --BS 1100 --Device cuda --imsize 256 --Split 0.9 --Epochs 7 --LR 1e-05 --ModelPath /mnt/GB480/facenet-pytorch-master_models/FaceNet__2024_10_08_15_13_36.pt --Resume

python Main.py TrainModel --data_dir /mnt/TB2/SynthPar2_Runs/Run_3/ --BS 1100 --Device cuda --imsize 256 --Split 0.9 --Epochs 7 --LR 1e-05 --ModelPath /mnt/GB480/facenet-pytorch-master_models/FaceNet__2024_10_08_15_13_36.pt --Resume

python Main.py TrainModel --data_dir /mnt/TB2/SynthPar2_Runs/Run_4/ --BS 1100 --Device cuda --imsize 256 --Split 0.9 --Epochs 7 --LR 1e-05 --ModelPath /mnt/GB480/facenet-pytorch-master_models/FaceNet__2024_10_08_15_13_36.pt --Resume

python Main.py TrainModel --data_dir /mnt/TB2/SynthPar2_Runs/Run_5/ --BS 1100 --Device cuda --imsize 256 --Split 0.9 --Epochs 7 --LR 1e-05 --ModelPath /mnt/GB480/facenet-pytorch-master_models/FaceNet__2024_10_08_15_13_36.pt --Resume

python Main.py TrainModel --data_dir /mnt/TB2/SynthPar2_Runs/Run_6/ --BS 1100 --Device cuda --imsize 256 --Split 0.9 --Epochs 7 --LR 1e-05 --ModelPath /mnt/GB480/facenet-pytorch-master_models/FaceNet__2024_10_08_15_13_36.pt --Resume

python Main.py TrainModel --data_dir /mnt/TB2/SynthPar2_Runs/Run_7/ --BS 1100 --Device cuda --imsize 256 --Split 0.9 --Epochs 7 --LR 1e-05 --ModelPath /mnt/GB480/facenet-pytorch-master_models/FaceNet__2024_10_08_15_13_36.pt --Resume

python Main.py TrainModel --data_dir /mnt/TB2/SynthPar2_Runs/Run_8/ --BS 1100 --Device cuda --imsize 256 --Split 0.9 --Epochs 7 --LR 1e-05 --ModelPath /mnt/GB480/facenet-pytorch-master_models/FaceNet__2024_10_08_15_13_36.pt --Resume


