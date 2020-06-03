import torch
import torch.nn as nn

class loss_collection(object):
    def __init__(self, cuda):
        self.cuda = cuda

    def construct_loss(self, mode='loglike'):
        if mode == 'loglike':
            return self.Gaussion_loglikehood
        elif mode == 'MSE':
            return self.MSE_loss
        else:
            raise NotImplementedError

    def Gaussion_loglikehood(self, logit, target):
        batch_size, coord_num, vector_num = logit.size()
        var = torch.var(logit)

        m = coord_num*vector_num

        criterion = *torch.log(2*3.14)/2 + 

        return 

    def MSE_loss(self, logit, target):
        criterion = nn.MSELoss(reduction='mean')

        if(self.cuda):
            criterion = criterion.cuda()
        
        loss = criterion(logit, target.long())

        return loss
        