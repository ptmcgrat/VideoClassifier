import argparse, subprocess, datetime, os, pdb, sys
from Utils.CichlidActionRecognition import ML_model


parser = argparse.ArgumentParser(description='This script takes video clips and annotations, either train a model from scratch or finetune a model to work on the new animals not annotated')
# Input data
parser.add_argument('--Videos_directory', type = str, required = True, help = 'Name of directory that holds mp4 files')
parser.add_argument('--Videos_file', type = str, required = True, help = 'Csv file listing each video. Must contain three columns: VideoFile, Label, ProjectID')
parser.add_argument('--Purpose', type = str, default = 'denovo', choices = ['denovo', 'finetune'], help = 'Train model from scratch or finetune existing model')

# Finetune input data
parser.add_argument('--Finetune_data', type = str, required = False, help = 'Csv file listing each video. Must contain three columns: VideoFile, Label, ProjectID')
parser.add_argument('--Trained_model', type=str,help='Save data (.pth) of previous training')
parser.add_argument('--Training_log', type = str, required = False, help = 'Necessary for finetuning. Lists parameters used in previous training')

# Output data
parser.add_argument('--Results_directory', type = str, required = True, help = 'Directory to store output files')

# Dataloader options
parser.add_argument('--n_threads', default=5, type=int, help='Number of threads for multi-thread loading')
parser.add_argument('--batch_size', default=13, type=int, help='Batch Size')
parser.add_argument('--gpu', default='0', type=str, help='The index of GPU to use for training')
parser.add_argument('--xy_crop', default=120, type=int, help='Temporal duration of inputs')
parser.add_argument('--t_crop', default=96, type=int, help='Height and width of inputs')
parser.add_argument('--t_interval', default=1, type=int, help='Height and width of inputs')
parser.add_argument('--projectMeans', action = 'store_true', help='Height and width of inputs')
parser.set_defaults(projectMeans=False)


# Parameters for the optimizer
parser.add_argument('--learning_rate',default=0.1,type=float,help='Initial learning rate (divided by 10 while training by lr scheduler)')
parser.add_argument('--momentum', default=0.9, type=float, help='Momentum')
parser.add_argument('--dampening', default=0.9, type=float, help='dampening of SGD')
parser.add_argument('--weight_decay', default=1e-5, type=float, help='Weight Decay')
parser.add_argument('--nesterov', action='store_true', help='Nesterov momentum', default = False)
parser.add_argument('--optimizer',default='sgd',type=str,help='Currently only support SGD')
parser.add_argument('--lr_patience',default=10,type=int,help='Patience of LR scheduler. See documentation of ReduceLROnPlateau.')
parser.add_argument('--resnet_shortcut',default='B',help='Shortcut type of resnet (A | B)')

# Parameters for general training
parser.add_argument('--checkpoint',default=10,type=int,help='Trained model is saved at every this epochs.')

# Parameters specific for training from scratch
parser.add_argument('--n_classes',default=10,type=int)
parser.add_argument('--n_epochs',default=100,type=int,help='Number of total epochs to run')

# Parameters specific for finetuning for other animals

args = parser.parse_args()

# Make directory if necessary
if not os.path.exists(args.Results_directory):
    os.makedirs(args.Results_directory)

if args.Purpose == 'finetune':
	prev_commands = {}
	with open(args.Training_log) as f:
		for line in f:
			prev_commands[line.rstrip().split(': ')[0]] = line.rstrip().split(': ')[1]
	for parameter in ['xy_crop', 't_crop', 't_interval']:
		args[parameter] = prev_commands[parameter]
	pdb.set_trace()

with open(os.path.join(args.Results_directory, 'TrainingLog.txt'),'w') as f:
	for key, value in vars(args).items():
		print(key + ': ' + str(value), file = f)
	print('PythonVersion: ' + sys.version.replace('\n', ' '), file = f)
	import pandas as pd
	print('PandasVersion: ' + pd.__version__, file = f)
	import numpy as np
	print('NumpyVersion: ' + np.__version__, file = f)
	import torch
	print('pytorch: ' + torch.__version__, file = f)
	
os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu
learningObj = ML_model(args.Results_directory, args.Videos_directory, args.Videos_file, args.xy_crop, args.t_crop, args.t_interval, args.n_classes)
learningObj.createModel()
learningObj.splitData('Train')
learningObj.createDataLoaders(args.batch_size, args.n_threads, args.projectMeans)
learningObj.trainModel(args.n_epochs, args.nesterov, args.dampening, args.learning_rate, args.momentum, args.weight_decay, args.lr_patience)