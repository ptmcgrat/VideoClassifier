import os, random, torch, time, pdb

import pandas as pd
import numpy as np

from torch import nn
from torch import optim
from torch.autograd import Variable
from torch.optim import lr_scheduler

from Utils.model import resnet18
from Utils.DataLoader import JPGLoader
from Utils.utils import Logger,AverageMeter,calculate_accuracy

#import sys
#import json
#import numpy as np
#import torch
#import torchvision
#from torch import nn
#from torch import optim
#from torch.optim import lr_scheduler
#from torch.autograd import Variable
#import pdb
#import pandas as pd
#import time

#from Utils.model import resnet18
#from Utils.utils import Logger,AverageMeter, calculate_accuracy

#from Utils.data_loader import cichlids


class ML_model():
    def __init__(self, results_directory, clips_directory, labeled_clips_file, xy_crop, t_crop, t_interval, n_classes):
        self.results_directory = results_directory
        self.clips_directory = clips_directory
        self.clips_dt = pd.read_csv(labeled_clips_file).head(200)
        self.xy_crop = xy_crop
        self.t_crop = t_crop
        self.t_interval = t_interval
        self.t_size = int(t_crop/t_interval)
        self.n_classes = n_classes
        #prepare the data is the data is not prepared
        #self.source_json_file = os.path.join(args.Results_directory,'source.json')

    def createModel(self):
        print('Creating Model')
        self.model = resnet18(
                num_classes=self.n_classes,
                shortcut_type='B',
                sample_size=self.xy_crop,
                sample_duration=self.t_size)

        self.model = self.model.cuda()
        self.model = nn.DataParallel(self.model, device_ids=None)
        self.parameters = self.model.parameters()
        self.criterion = nn.CrossEntropyLoss()
        self.criterion = self.criterion.cuda()

    def splitData(self, analysis_type):
        print('Splitting Data')

        self.clips_dt['Dataset'] = ''
        if analysis_type == 'Train':
            train_cutoff = 0.8
            val_cutoff = 1.0
        elif analysis_type == 'Predict':
            train_cutoff = 0
            val_cutoff = 1.0
        else:
            pass

        for index, row in self.clips_dt.iterrows():
            p = random.random()
            if p<=train_cutoff:
                self.clips_dt.at[index,'Dataset'] = 'Train'
            elif p<=val_cutoff:
                self.clips_dt.at[index,'Dataset'] = 'Validate'
            else:
                self.clips_dt.at[index,'Dataset'] = 'Test'

    def createDataLoaders(self, batch_size, n_threads):
        self.batch_size = batch_size
        self.n_threads = n_threads
        print('Creating Data Loaders')

        trainData = JPGLoader(self.clips_directory, self.clips_dt[self.clips_dt.Dataset == 'Train'], self.xy_crop, self.t_crop, self.t_interval, augment = True, projectMeans = True)
        valData = JPGLoader(self.clips_directory, self.clips_dt[self.clips_dt.Dataset == 'Validate'], self.xy_crop, self.t_crop, self.t_interval, augment = True, projectMeans = True)

        self.trainLoader = torch.utils.data.DataLoader(trainData, batch_size = self.batch_size, shuffle = True, num_workers = self.n_threads, pin_memory = True)
        self.valLoader = torch.utils.data.DataLoader(valData, batch_size = self.batch_size, shuffle = False, num_workers = self.n_threads, pin_memory = True)
        print('Done')

    def trainModel(self, n_epochs, nesterov, dampening, learning_rate, momentum, weight_decay, lr_patience):
        
        train_logger = Logger(os.path.join(self.results_directory, 'train.log'), ['epoch', 'loss', 'acc', 'lr'])
        train_batch_logger = Logger(os.path.join(self.results_directory, 'train_batch.log'), ['epoch', 'batch', 'iter', 'loss', 'acc', 'lr'])
        val_logger = Logger(os.path.join(self.results_directory, 'val.log'), ['epoch', 'loss', 'acc'])

        if nesterov:
            dampening = 0
        else:
            dampening = dampening
       
        optimizer = optim.SGD(self.parameters, lr=learning_rate, momentum=momentum, dampening=dampening, weight_decay=weight_decay, nesterov=nesterov)
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=lr_patience)

        for i in range(n_epochs + 1):
            self.train_epoch(i, self.trainLoader, self.model, self.criterion, optimizer, train_logger, train_batch_logger)

            validation_loss,confusion_matrix,_ = self.val_epoch(i, self.valLoader, self.model, self.criterion, val_logger)

            confusion_matrix_file = os.path.join(self.results_directory,'epoch_{epoch}_confusion_matrix.csv'.format(epoch=i))
            confusion_matrix.to_csv(confusion_matrix_file)

            scheduler.step(validation_loss)
            #if i % 5 == 0 and len(test_data) != 0:
            #   _ = self.val_epoch(i, test_loader, model, criterion, opt, test_logger)

    def train_epoch(self, epoch, data_loader, model, criterion, optimizer,
                    epoch_logger, batch_logger):
        print('train at epoch {}'.format(epoch))
        model.train()

        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        accuracies = AverageMeter()

        end_time = time.time()
        for i, (inputs, targets, paths) in enumerate(data_loader):
            data_time.update(time.time() - end_time)

            targets = targets.cuda(non_blocking=True)
            inputs = Variable(inputs)
            targets = Variable(targets)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            acc = calculate_accuracy(outputs, targets)

            losses.update(loss.data, inputs.size(0))
            accuracies.update(acc, inputs.size(0))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_time.update(time.time() - end_time)
            end_time = time.time()

            batch_logger.log({
                'epoch': epoch,
                'batch': i + 1,
                'iter': (epoch - 1) * len(data_loader) + (i + 1),
                'loss': losses.val,
                'acc': accuracies.val,
                'lr': optimizer.param_groups[0]['lr']
            })

            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Acc {acc.val:.3f} ({acc.avg:.3f})'.format(
                epoch,
                i + 1,
                len(data_loader),
                batch_time=batch_time,
                data_time=data_time,
                loss=losses,
                acc=accuracies))
        epoch_logger.log({
                'epoch': epoch,
                'loss': losses.avg,
                'acc': accuracies.avg,
                'lr': optimizer.param_groups[0]['lr']
            })

        if epoch % 5 == 0:
            save_file_path = os.path.join(self.results_directory,
                                          'save_{}.pth'.format(epoch))
            states = {
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }
            torch.save(states, save_file_path)

    def val_epoch(self, epoch, data_loader, model, criterion, logger):
        print('validation at epoch {}'.format(epoch))

        model.eval()

        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        accuracies = AverageMeter()

        end_time = time.time()
        confusion_matrix = np.zeros((self.n_classes,self.n_classes))
        confidence_for_each_validation = {}
        ###########################################################################

        # pdb.set_trace()
        for i, (inputs, targets, paths) in enumerate(data_loader):
            data_time.update(time.time() - end_time)

            targets = targets.cuda(non_blocking=True)
            with torch.no_grad():
                inputs = Variable(inputs)
                targets = Variable(targets)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                acc = calculate_accuracy(outputs, targets)
                ########  temp line, needs to be removed##################################
                for j in range(len(targets)):
                    key = paths[j].split('/')[-1]
                    confidence_for_each_validation[key] = [x.item() for x in outputs[j]]

                rows = [int(x) for x in targets]
                columns = [int(x) for x in np.argmax(outputs.data.cpu(),1)]
                assert len(rows) == len(columns)
                for idx in range(len(rows)):
                    confusion_matrix[rows[idx]][columns[idx]] +=1

                ###########################################################################
                losses.update(loss.data, inputs.size(0))
                accuracies.update(acc, inputs.size(0))

                batch_time.update(time.time() - end_time)
                end_time = time.time()

                print('Epoch: [{0}][{1}/{2}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Acc {acc.val:.3f} ({acc.avg:.3f})'.format(
                    epoch,
                    i + 1,
                    len(data_loader),
                    batch_time=batch_time,
                    data_time=data_time,
                    loss=losses,
                    acc=accuracies))
            #########  temp line, needs to be removed##################################
            # print(confusion_matrix)
        confusion_matrix = pd.DataFrame(confusion_matrix)
            # confusion_matrix.to_csv(file)
        confidence_matrix = pd.DataFrame.from_dict(confidence_for_each_validation, orient='index')
        
            #     confidence_matrix.to_csv('confidence_matrix.csv')

            #########  temp line, needs to be removed##################################

        logger.log({'epoch': epoch, 'loss': losses.avg, 'acc': accuracies.avg})

        return losses.avg,confusion_matrix,confidence_matrix
        
    def test_epoch(self, epoch, data_loader, model, criterion, logger):
        print('test at epoch {}'.format(epoch))

        model.eval()

        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        accuracies = AverageMeter()

        end_time = time.time()

        for i, (inputs, targets,_) in enumerate(data_loader):
            data_time.update(time.time() - end_time)
            if not opt.no_cuda:
                targets = targets.cuda(non_blocking=True)
                with torch.no_grad():
                    inputs = Variable(inputs)
                    targets = Variable(targets)
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                    acc = calculate_accuracy(outputs, targets)
                    losses.update(loss.data, inputs.size(0))
                    accuracies.update(acc, inputs.size(0))

                    batch_time.update(time.time() - end_time)
                    end_time = time.time()

                    print('Epoch: [{0}][{1}/{2}]\t'
                          'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                          'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                          'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                          'Acc {acc.val:.3f} ({acc.avg:.3f})'.format(
                        epoch,
                        i + 1,
                        len(data_loader),
                        batch_time=batch_time,
                        data_time=data_time,
                        loss=losses,
                        acc=accuracies))
            logger.log({'epoch': epoch, 'loss': losses.avg, 'acc': accuracies.avg})

            return losses.avg
    