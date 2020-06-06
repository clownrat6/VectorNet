import os
import cv2
import torch
import torch.nn as nn
import numpy as np

from tqdm import tqdm
from arg_parser import arg_parser
from modeling.VectorNet import *
from modeling.padding_VectorNet import *
from utils.losses import loss_collection
from utils.lr_scheduler import LR_Scheduler
from utils.metricer import metricer

class train:
    def __init__(self, args):
        self.args = args

        kwargs = {'num_workers': args.workers, 'pin_memory': True}

        # Define Optimizer,model
        if(args.model == 'padding_vectornet'):
            model = padding_VectorNet(args.depth_sub, args.width_sub, args.depth_global, args.width_global)
            train_params = [{'params': model.get_1x_lr_params(), 'lr': args.lr}]
        elif(args.model == 'vectornet'):
            model = VectorNet(args.depth_sub, args.width_sub, args.depth_global, args.width_global)
            train_params = [{'params': model.get_1x_lr_params(), 'lr': args.lr}]
        else:
            assert False, 'Error!!\nUnsupported model: {}'.format(args.model)

        # CUDA enabled
        if(args.cuda):
            self.model = self.model.cuda()

        optimizer = torch.optim.Adam(train_params)
        self.criterion = loss_collection().construct_loss(args.loss_mode)

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

        pass

    def training(self, epoch):
        train_loss = 0.0
        # train() function can activate BN layer and Dropout layer
        self.model.train()
        pass

    def validating(self, epoch):
        pass

    def only_validation(self):
        pass

if __name__ == '__main__':
    parser = arg_parser()

    parser.add_val('--workers', 2)
    parser.add_val('--model', 'vectornet')
    parser.add_val('--dataset', 'Argoverse')
    parser.add_val('--dataset_dir', r'data')
    parser.add_val('--batch_size', 2)
    parser.add_val('--cuda', True)
    parser.add_val('--loss_mode', 'ce')
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