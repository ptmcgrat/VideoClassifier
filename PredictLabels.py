import argparse, subprocess, datetime, os, pdb, sys
from Utils.CichlidActionRecognition import ML_model


parser = argparse.ArgumentParser(description='This script takes video clips and annotations, either train a model from scratch or finetune a model to work on the new animals not annotated')
# Input data
parser.add_argument('--Videos_directory', type = str, required = True, help = 'Name of directory that holds mp4 files')
parser.add_argument('--Videos_file', type = str, required = True, help = 'Csv file listing each video. Must contain three columns: VideoFile, Label, ProjectID, ModelOrigin. Dataset is Original|New')

# Finetune input data
parser.add_argument('--Trained_model', type=str, required = False, help='Save data (.pth) of previous training')
parser.add_argument('--Training_log', type = str, required = False, help = 'Necessary for finetuning. Lists parameters used in previous training')

# Output data
parser.add_argument('--Results_directory', type = str, required = True, help = 'Directory to store output files')

# Dataloader options
parser.add_argument('--n_threads', default=5, type=int, help='Number of threads for multi-thread loading')
parser.add_argument('--batch_size', default=13, type=int, help='Batch Size')
parser.add_argument('--gpu', default='0', type=str, help='The index of GPU to use for training')

args = parser.parse_args()

# Make directory if necessary
if not os.path.exists(args.Results_directory):
    os.makedirs(args.Results_directory)

# Read in previous commands
prev_commands = {}
with open(args.Training_log) as f:
	for line in f:
		prev_commands[line.rstrip().split(': ')[0]] = line.rstrip().split(': ')[1]
for parameter in ['xy_crop','t_crop','t_interval','projectMeans','learning_rate','momentum','dampening','weight_decay','nesterov','optimizer','lr_patience','n_classes']:
	args.__dict__[parameter] = int(prev_commands[parameter])

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
learningObj.splitData('Predict')
learningObj.createDataLoaders(args.batch_size, args.n_threads, args.projectMeans)
learningObj.trainModel(args.n_epochs, args.nesterov, args.dampening, args.learning_rate, args.momentum, args.weight_decay, args.lr_patience, args.Trained_model)