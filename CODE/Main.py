# conda activate facenet
import argparse, datetime, LIB_FaceNet

def main():
	parser = argparse.ArgumentParser("FaceNet Bias Library");
	subparsers = parser.add_subparsers();


	subparser = subparsers.add_parser("TrainModel", description = "Train FaceNet model");
	subparser.add_argument("--data_dir", required = True, type = str, help = "Dataset location");
	subparser.add_argument("--BS", default = 32, type = int, help = "Batch size");
	subparser.add_argument("--LR", default = 0.001, type = float, help = "Learning Rate");
	subparser.add_argument("--Split", default = 0.8, type = float, help = "Train/Validate split");	
	subparser.add_argument("--imsize", default = 512, type = int, help = "Image resize samples");
	subparser.add_argument("--Epochs", default = 4, type = int, help = "Number of epochs");
	subparser.add_argument("--Device", default = "cuda:0", type = str, help = "Device");
	subparser.add_argument("--Workers", default = 6, type = int, help = "Number of CPU workers");
	subparser.add_argument("--ModelPath", default = "", type = str, help = "Path to save trained models");		
	subparser.add_argument("--Resume", action="store_true", help = "Verbose (False) resume and overwrite model");
	subparser.add_argument("--TFWritter", action="store_true", help = "Tensor Flow log writer");
	subparser.add_argument("--Visualize", action="store_true", help = "Visualize (False)");			
	subparser.set_defaults(func = LIB_FaceNet.TrainModel);


	subparser = subparsers.add_parser("TestModel", description = "Test FaceNet model");
	subparser.add_argument("--ModelPath", required = True, type = str, help = "Path to load trained model");		
	subparser.add_argument("--data_dir", required = True, type = str, help = "Dataset location");
	subparser.add_argument("--BS", default = 32, type = int, help = "Batch size");
	subparser.add_argument("--imsize", default = 512, type = int, help = "Image resize samples");
	subparser.add_argument("--Device", default = "cuda:0", type = str, help = "Device");
	subparser.add_argument("--Workers", default = 6, type = int, help = "Number of CPU workers");
	subparser.add_argument("--Stage", default = "test", type = str, help = "Device");	
	subparser.set_defaults(func = LIB_FaceNet.TestModel);

	Options = parser.parse_args();

	print(str(Options) + "\n");

	Response = Options.func(Options);



if __name__ == "__main__":
	print("\n" + "\033[0;32m" + "[start] " + str(datetime.datetime.now()) + "\033[0m" + "\n");
	main();
	print("\n" + "\033[0;32m" + "[end] "+ str(datetime.datetime.now()) + "\033[0m" + "\n");
