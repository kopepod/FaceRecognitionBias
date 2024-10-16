import torch, torchvision, numpy, os, datetime
from facenet_pytorch import MTCNN, InceptionResnetV1, fixed_image_standardization
from models.utils import training

def datetimestring():
	now = datetime.datetime.now()
	return now.strftime("%Y_%m_%d_%H_%M_%S")	
	
	
def LoadModel(Options, num_classes):

	device = torch.device(Options.Device if torch.cuda.is_available() else 'cpu');

	if Options.Resume:
		#
		resnet = InceptionResnetV1(
    classify = True,
    pretrained = None,
    num_classes = num_classes)
    
		check_point = torch.load(Options.ModelPath);
		resnet.load_state_dict(check_point)

	else:
		#
		resnet = InceptionResnetV1(
    	classify = True,
    	pretrained = 'vggface2',
    	num_classes = num_classes)


	resnet = resnet.to(device);

	if ":" not in Options.Device:
		resnet = torch.nn.DataParallel(resnet);
		print("\n" + "\033[1;33m" + "Using multi-GPU support" + "\033[0m" + "\n");
	else:
		print("\n" + "\033[1;33m" + "Single GPU " + Options.Device + "\033[0m" + "\n");	
		
	return resnet;

def LoadDataset(Options, Stage):

	batch_size = Options.BS;
	workers = 0 if os.name == 'nt' else Options.Workers;
	
	trans = torchvision.transforms.Compose([
    torchvision.transforms.Resize( (Options.imsize, Options.imsize) ),
    torchvision.transforms.ToTensor(),
    fixed_image_standardization])

	dataset = torchvision.datasets.ImageFolder(Options.data_dir, transform = trans);

	num_classes = len(dataset.class_to_idx);
	
	print("Number of classes : ", num_classes);
	
	match Stage:
		case "train":
			img_inds = numpy.arange(len(dataset))
			numpy.random.shuffle(img_inds)
			train_inds = img_inds[:int(Options.Split * len(img_inds))]
			val_inds = img_inds[int(Options.Split * len(img_inds)):]
			#
			train_loader = torch.utils.data.DataLoader(
  		  dataset, 
  		  num_workers = workers,
  		  batch_size = batch_size,
  		  sampler = torch.utils.data.SubsetRandomSampler(train_inds))
			val_loader = torch.utils.data.DataLoader(
  		  dataset,
  		  num_workers = workers,
  		  batch_size = batch_size // 2,
  		  sampler = torch.utils.data.SubsetRandomSampler(val_inds))
			#
			return (train_loader, val_loader, num_classes)
		#
		case "test":
			test_loader = torch.utils.data.DataLoader(dataset, num_workers = workers,  batch_size = batch_size //2)
			return (test_loader, num_classes)
		case _ :
			print("Invalid stage")
			return None;
	
def TrainModel(Options):

	device = torch.device(Options.Device if torch.cuda.is_available() else 'cpu');
	
	train_loader, val_loader, num_classes = LoadDataset(Options, "train");
	
	resnet = LoadModel(Options, num_classes);
		
	optimizer = torch.optim.Adam(resnet.parameters(), lr = Options.LR);
	scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [5, 10]);
	loss_fn = torch.nn.CrossEntropyLoss();
	
	metrics = {
    'fps': training.BatchTimer(),
    'acc': training.accuracy
	}

	if Options.TFWritter:
		writer = torch.utils.tensorboard.SummaryWriter()
		writer.iteration, writer.interval = 0, 10
	else:
		writer = None;

	print('Training')
	print('_' * 10)

	for epoch in range(Options.Epochs):
		print('\nEpoch {}/{}'.format(epoch + 1, Options.Epochs));
		print('-' * 10);

		resnet.train();
		training.pass_epoch(
        resnet, loss_fn, train_loader, optimizer, scheduler,
        batch_metrics = metrics, show_running = True, device = device,
        writer=writer
		)

		resnet.eval();
		training.pass_epoch(
        resnet, loss_fn, val_loader,
        batch_metrics = metrics, show_running = True, device = device,
        writer=writer
		)


	if Options.TFWritter:
		writer.close()

	if Options.ModelPath == "":
		print("Not saving model ...");
	else:
		if Options.Resume:
			FileName = Options.ModelPath;
		else:
			FileName = Options.ModelPath + "FaceNet__" + datetimestring() + ".pt";
		
		print("Saving model : %s" %(FileName));

		if ":" not in Options.Device:
			torch.save(resnet.module.state_dict(), FileName);
		else:
			torch.save(resnet.state_dict(), FileName);

		

def TestModel(Options):

	Options.Resume = True;

	device = torch.device(Options.Device if torch.cuda.is_available() else 'cpu');
	
	test_loader, num_classes = LoadDataset(Options, "test");
	
	resnet = LoadModel(Options, num_classes);	
	
	loss_fn = torch.nn.CrossEntropyLoss();
	
	metrics = {
    'fps': training.BatchTimer(),
    'acc': training.accuracy
	}

	print('Testing')
	print('_' * 10)

	resnet.eval();
	training.pass_epoch(
        resnet, loss_fn, test_loader,
        batch_metrics = metrics, show_running = True, device = device,
        writer=None	)
































		
