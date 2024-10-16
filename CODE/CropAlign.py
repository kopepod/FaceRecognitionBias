from facenet_pytorch import MTCNN, InceptionResnetV1, fixed_image_standardization, training
import torch
from torch.utils.data import DataLoader, SubsetRandomSampler
from torch import optim
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms
import numpy as np
import os

data_dir = '/mnt/TB2/DATASETS/VGGFace2'

batch_size = 32
epochs = 8
workers = 0 if os.name == 'nt' else 8

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('Running on device: {}'.format(device))


mtcnn = MTCNN(
	image_size=160, margin=0, min_face_size=20,
	thresholds=[0.6, 0.7, 0.7], factor=0.709, post_process=True,
	device=device
)

dataset = datasets.ImageFolder(data_dir, transform=transforms.Resize((512, 512)))
dataset.samples = [
	(p, p.replace(data_dir, data_dir + '_cropped'))
		for p, _ in dataset.samples
]
		
loader = DataLoader(
	dataset,
	num_workers=workers,
	batch_size=batch_size,
	collate_fn=training.collate_pil
)

for i, (x, y) in enumerate(loader):
	try:
		mtcnn(x, save_path=y)
		print('\rBatch {} of {}'.format(i + 1, len(loader)), end='')
	except:
		continue
	
# Remove mtcnn to reduce GPU memory usage
del mtcnn
