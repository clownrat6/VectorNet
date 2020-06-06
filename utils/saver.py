import os
import torch

class saver(object):
    def __init__(self, save_base_path, model_name):
        self.model_base_path = save_base_path
        self.best_pred = 0
        self.epoch = 0

    def restore(self):
        model_list = os.listdir(self.model_base_path)
        if(len(model_list) == 0): 
            print('=>{}, There is no checkpoint'.format(self.model_base_path))
            return None
        else:
            model_list = sorted(model_list, key=lambda x:int(os.path.splitext(x)[0].split('_')[0]))
            self.selected_model = model_list[-1]
            self.best_pred = float(os.path.splitext(self.selected_model)[0].split('_')[1])
            self.selected_model_path = os.path.join(self.model_base_path, model_list[-1])
            print('=> loaded checkpoint {}'.format(self.selected_model))
            checkpoint = torch.load(self.selected_model_path)
            self.best_pred = checkpoint['best_pred']
            self.epoch = checkpoint['epoch']
            return checkpoint['model'],checkpoint['optimizer'],self.epoch,self.best_pred

    def save(self, epoch, pred, model, optimizer):
        if(pred > self.best_pred):
            model_save_path = os.path.join(self.model_base_path, '{}_%.4f.pth'.format(epoch)%(pred))
            torch.save(
                {'model': model.state_dict(),
                 'optimizer': optimizer.state_dict(),
                 'best_pred': pred, # validation mIoU, we should make sure the optimize target is mIoU.
                 'epoch':epoch+1}, model_save_path)
            self.best_pred = pred
            return '{} >= {}, model is saved to {}'.format(pred, pred, model_save_path)
        else:
            return "{} < {}, model isn't saved."