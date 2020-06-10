import os
import cv2
import torch
import torch.nn as nn
import numpy as np

from arg_parser import arg_parser
from dataloader import construct_loader
from modeling.VectorNet import *
from modeling.padding_VectorNet import *
from utils.losses import loss_collection
from utils.lr_scheduler import LR_Scheduler
from utils.metricer import metricer
from utils.logger import logger
from utils.saver import saver

class train:
    def __init__(self, args):
        self.args = args

        self.train_loader, self.valid_loader = construct_loader(args.train_path, \
            args.valid_path, args.batch_size, args.dataset, args.cuda)

        # Define Optimizer,model
        if(args.model == 'padding_vectornet'):
            model = padding_VectorNet(args.depth_sub, args.width_sub, args.depth_global, args.width_global)
            train_params = [{'params': model.parameters(), 'lr': args.lr}]
        elif(args.model == 'vectornet'):
            model = VectorNet(args.depth_sub, args.width_sub, args.depth_global, args.width_global)
            train_params = [{'params': model.parameters(), 'lr': args.lr}]
        else:
            assert False, 'Error!!\nUnsupported model: {}'.format(args.model)

        self.model = model

        # CUDA enabled
        if(args.cuda):
            self.model = self.model.cuda()

        self.optimizer = torch.optim.Adam(train_params)
        self.criterion = loss_collection(args.cuda).construct_loss(args.loss_mode)

        # loss weight selection
        self.scheduler = LR_Scheduler(args.lr_scheduler, args.lr,
                                    args.epochs, len(self.train_loader))

        self.metricer = metricer()

        if(not os.path.exists('ckpt/{}'.format(args.model))):
            os.makedirs('ckpt/{}'.format(args.model), 0o777)
        self.logger = logger('ckpt/{}'.format(args.model), ['DE@1s', 'DE@2s', 'DE@3s', 'ADE', 'loss'])
        if(not os.path.exists('ckpt/{}/storage'.format(args.model))):
            os.makedirs('ckpt/{}/storage'.format(args.model), 0o777)
        self.saver = saver('ckpt/{}/storage'.format(args.model), args.model)
        ret = self.saver.restore()
        self.start_epoch = 1
        self.best_pred = 0
        if(ret != None):
            self.model.load_state_dict(ret[0])
            self.optimizer.load_state_dict(ret[1])
            self.start_epoch = ret[2]
            self.best_pred = ret[3]


    def training(self, epoch):
        train_loss = 0.0
        # train() function can activate BN layer and Dropout layer
        self.model.train()
        if(self.args.padding):
            generator = self.train_loader.padding_dataset_generator()
        else:
            generator = self.train_loader.dataset_generator()

        i = 0
        for map_pres, train_traj, test_traj in generator:
            i += 1
            self.scheduler(self.optimizer, i+1, epoch, self.best_pred)
            self.optimizer.zero_grad()
            pred_traj = self.model(train_traj, map_pres)
            loss = self.criterion(pred_traj, test_traj)
            loss.backward()
            self.optimizer.step()
            train_loss += loss.item()
            temp_string = 'Train loss: %.3f {}'.format(self.optimizer.param_groups[0]['lr']) % (train_loss / i)
            print(temp_string)
            self.logger.detail_write(temp_string)
            

        temp_string = '[Epoch: %d, Batches: %d] training\n' % (epoch, self.args.batch_size * i)
        temp_string += 'Average Loss: %.3f' % (train_loss)
        print(temp_string)
        self.logger.detail_write(temp_string)

    def validating(self, epoch):
        self.model.eval()
        if(self.args.padding):
            generator = self.train_loader.padding_dataset_generator()
        else:
            generator = self.train_loader.dataset_generator()
        
        valid_loss = 0.0
        i = 0
        for map_pres, train_traj, test_traj in generator:
            i += 1
            with torch.no_grad():
                pred_traj = self.model(train_traj, map_pres)
            loss = self.criterion(pred_traj, test_traj)
            valid_loss += loss.item()
            temp_string = 'Test loss: %.3f' % (valid_loss / i)
            print(temp_string)
            self.metricer.add_batch(pred_traj, test_traj)

        DE1s, DE2s, DE3s, ADE = self.metricer.calculate()
        temp_string = '[Epoch: %d, Batches: %d] validation\n' % (epoch, self.args.batch_size * i)
        temp_string += 'Average Loss: %.3f' % (valid_loss)
        temp_string += 'metric -- DE@1s:{} DE@2s:{} DE@3s:{} ADE:{}'.format(DE1s, DE2s, DE3s, ADE)
        print(temp_string)
        if(ADE < self.best_pred):
            self.best_pred = ADE
        
        self.logger.detail_write(temp_string)
        self.logger.target_save(epoch, 'DE@1s', DE1s);self.logger.target_save(epoch, 'DE@2s', DE2s)
        self.logger.target_save(epoch, 'DE@3s', DE3s);self.logger.target_save(epoch, 'ADE', ADE)
        self.saver.save(epoch, ADE, self.model, self.optimizer)

if __name__ == '__main__':
    parser = arg_parser()

    parser.add_val('--workers', 2)
    parser.add_val('--model', 'padding_vectornet')
    parser.add_val('--padding', True)
    parser.add_val('--dataset', 'Argoverse')
    parser.add_val('--depth_sub', 3)
    parser.add_val('--width_sub', 64)
    parser.add_val('--depth_global', 1)
    parser.add_val('--width_global', 128)
    parser.add_val('--train_path', r'data/argoverse-forecasting/forecasting_train_v1.1/data')
    parser.add_val('--valid_path', r'data/argoverse-forecasting/forecasting_val_v1.1/data')
    parser.add_val('--batch_size', 2)
    parser.add_val('--cuda', True)
    parser.add_val('--loss_mode', 'loglike')
    parser.add_val('--lr_scheduler', 'poly')
    parser.add_val('--lr', 0.0001)
    parser.add_val('--epochs', 25)
    parser.add_val('--no_val', False)
    parser.add_val('--validation_interval', 1)

    do = train(parser)
    args = parser

    for epoch in range(do.start_epoch, args.epochs):
        do.training(epoch)
        if((epoch+1)%args.validation_interval == 0 and not args.no_val):
            do.validating(epoch)