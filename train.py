import os
import cv2
import torch
import torch.nn as nn
import numpy as np

from tqdm import tqdm
from dataloaders import construct_loader
from arg_parser import arg_parser
from model.deeplab import *
from model.attention_unet import *
from model.unet import *
from model.danet import *
from utils.losses import loss_collection
from utils.metrics import metricer
from utils.lr_scheduler import LR_Scheduler
from utils.logger import logger
from utils.saver import saver

import matplotlib.pyplot as plt

class train:
    def __init__(self, args):
        self.args = args

        kwargs = {'num_workers': args.workers, 'pin_memory': True}
        self.train_loader,self.valid_loader,num_class = \
            construct_loader(args.dataset, args.dataset_dir, args.batch_size, args.base_size, args.crop_size, **kwargs)

        # Define Optimizer,model
        if(args.model == 'deeplab'):
            model = deeplabv3plus('resnet', 16, num_class)
            train_params = [{'params': model.get_1x_lr_params(), 'lr': args.lr},
                            {'params': model.get_10x_lr_params(), 'lr': args.lr * 10}]
        elif(args.model == 'attunet'):
            model = attention_unet(3, 64, num_class)
            train_params = [{'params': model.parameters(), 'lr': args.lr}]
        elif(args.model == 'unet'):
            model = unet(3, 64, num_class)
            train_params = [{'params': model.parameters(), 'lr': args.lr}]
        elif(args.model == 'danet'):
            model = danet(num_class)
            train_params = [{'params': model.parameters(), 'lr': args.lr}]
        else:
            assert False, 'Error!!\nUnsupported model: {}'.format(args.model)

        
        optimizer = torch.optim.SGD(train_params, momentum=0.9, weight_decay=5e-4, nesterov=False)
        
        # loss weight selection
        self.criterion = loss_collection(weight=None, cuda=args.cuda).construct_loss(args.loss_mode)
        self.model,self.optimizer = model,optimizer

        self.evaluator = metricer(num_class)
        self.scheduler = LR_Scheduler(args.lr_scheduler, args.lr,
                                    args.epochs, len(self.train_loader))

        # CUDA enabled
        if(args.cuda):
            self.model = self.model.cuda()

        # restoring checkpoint or starting
        if(not os.path.exists('ckpt/{}'.format(args.model))):
            os.makedirs('ckpt/{}'.format(args.model), 0o777)
        self.logger = logger('ckpt/{}'.format(args.model), ['mIoU', 'mDSC', 'loss'])
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
        tbar = tqdm(self.train_loader)
        num_img_tr = len(self.train_loader)
        for i, sample in enumerate(tbar):
            image, target = sample['image'], sample['label']
            if self.args.cuda:
                image, target = image.cuda(), target.cuda()
            self.scheduler(self.optimizer, i+1, epoch, self.best_pred)
            self.optimizer.zero_grad()
            output = self.model(image)
            # plt.subplot(131)
            # plt.imshow(image.cpu().numpy()[0].transpose(1, 2, 0))
            # plt.subplot(132)
            # print(output.size())
            # plt.imshow(np.argmax(output.cpu().detach().numpy()[0], axis=0))
            # plt.subplot(133)
            # plt.imshow(target.cpu().numpy()[0])
            # plt.show()
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()
            train_loss += loss.item()
            tbar.set_description('Train loss: %.3f {}'.format(self.optimizer.param_groups[0]['lr'])% (train_loss / (i + 1)))
            temp_string = 'Train loss: %.3f {}'.format(self.optimizer.param_groups[0]['lr']) % (train_loss / (i + 1))
            self.logger.detail_write(temp_string)
            temp_string += '\n'
            # Show 10 * 3 inference results each epoch
            if i % (num_img_tr // 10) == 0:
                global_step = i + num_img_tr * epoch

        temp_string += '[Epoch: %d, numImages: %5d]\n' % (epoch, i * self.args.batch_size + image.data.shape[0])
        temp_string += 'Loss: %.3f' % train_loss
        self.logger.detail_write(temp_string)
        print(temp_string)

    def validating(self, epoch):
        self.model.eval()
        self.evaluator.reset()
        valid_loss = 0.0
        tbar = tqdm(self.valid_loader, desc='\r')
        for i, sample in enumerate(tbar):
            image, target = sample['image'], sample['label']
            if self.args.cuda:
                image, target = image.cuda(), target.cuda()
            with torch.no_grad():
                output = self.model(image)
            loss = self.criterion(output, target)
            valid_loss += loss.item()
            tbar.set_description('Test loss: %.3f' % (valid_loss / (i + 1)))
            pred = output.data.cpu().numpy()
            target = target.cpu().numpy()
            pred = np.argmax(pred, axis=1)
            # plt.subplot(131)
            # plt.imshow(pred[1])
            # plt.subplot(132)
            # plt.imshow(target[1])
            # plt.subplot(133)
            # plt.imshow(image.cpu().numpy()[1].transpose(1, 2, 0))
            # plt.show()
            # Add batch sample into evaluator
            self.evaluator.add_batch(pred, target)

        # Fast test during the training
        mIoU = self.evaluator.mean_Intersection_over_Union()
        mDSC = self.evaluator.mean_Dice_Similarity_Coefficient()
        temp_string = 'Validation:\n'
        temp_string += '[Epoch: %d, numImages: %5d]\n' % (epoch+1, i * self.args.batch_size + image.data.shape[0])
        temp_string += "mIoU:{}, mDSC:{} ".format(mIoU, mDSC)
        temp_string += 'Loss: %.3f' % (valid_loss)
        if(mDSC > self.best_pred):
            self.best_pred = mDSC
        print(temp_string)
        self.logger.detail_write(temp_string)
        self.logger.target_save(epoch+1, 'mIoU', mIoU);self.logger.target_save(epoch+1, 'mDSC', mDSC);self.logger.target_save(epoch+1, 'loss', valid_loss)
        self.saver.save(epoch, mDSC, self.model, self.optimizer)

    def only_validation(self):
        self.model.eval()
        self.evaluator.reset()
        valid_loss = 0.0
        tbar = tqdm(self.valid_loader, desc='\r')
        for i, sample in enumerate(tbar):
            image, target = sample['image'], sample['label']
            if self.args.cuda:
                image, target = image.cuda(), target.cuda()
            with torch.no_grad():
                output = self.model(image)
            loss = self.criterion(output, target)
            valid_loss += loss.item()
            tbar.set_description('Test loss: %.3f' % (valid_loss / (i + 1)))
            pred = output.data.cpu().numpy()
            target = target.cpu().numpy()
            pred = np.argmax(pred, axis=1)
            pred = self.postprocess(pred)
            # plt.subplot(131)
            # plt.imshow(pred[1])
            # plt.subplot(132)
            # plt.imshow(target[1])
            # plt.subplot(133)
            # plt.imshow(image.cpu().numpy()[1].transpose(1, 2, 0))
            # plt.show()
            # Add batch sample into evaluator
            self.evaluator.add_batch(pred, target)

        # Fast test during the training
        mIoU = self.evaluator.mean_Intersection_over_Union()
        mDSC = self.evaluator.mean_Dice_Similarity_Coefficient()
        temp_string = 'Validation:\n'
        temp_string += '[Test only, numImages: %5d]\n' % (i * self.args.batch_size + image.data.shape[0])
        temp_string += "mIoU:{}, mDSC:{} ".format(mIoU, mDSC)
        temp_string += 'Loss: %.3f' % (valid_loss)
        print(temp_string)

    def postprocess(self, masks):
        mask_list = []
        # Rect shape kernel
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
        for mask in masks:
            # print(mask.shape)
            mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, kernel)
            mask_list.append(mask)
        
        return np.stack(mask_list, axis=0)

if __name__ == '__main__':
    parser = arg_parser()

    parser.add_val('--workers', 2)
    parser.add_val('--model', 'deeplab')
    parser.add_val('--dataset', 'taidi')
    parser.add_val('--dataset_dir', r'E:\dataset')
    parser.add_val('--batch_size', 1)
    parser.add_val('--base_size', 1536)
    parser.add_val('--crop_size', 1536)
    parser.add_val('--cuda', True)
    parser.add_val('--loss_mode', 'ce')
    parser.add_val('--lr_scheduler', 'poly')
    parser.add_val('--lr', 0.007)
    parser.add_val('--epochs', 50)
    parser.add_val('--no_val', False)
    parser.add_val('--validation_interval', 1)

    do = train(parser)
    args = parser
    do.only_validation()

    # for epoch in range(do.start_epoch, args.epochs):
    #     do.training(epoch)
    #     if((epoch+1)%args.validation_interval == 0 and not args.no_val):
    #         do.validating(epoch)