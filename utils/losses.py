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

        Sigma = torch.sqrt(var)

        m = coord_num*vector_num

        loss = m*torch.log(torch.tensor(2*3.14))/2 + m*torch.log(Sigma) + \
            1/(2*var)*torch.sum(torch.square(logit-target)) 

        if(self.cuda):
            loss = loss.cuda()

        return loss

    def MSE_loss(self, logit, target):
        criterion = nn.MSELoss(reduction='mean')

        if(self.cuda):
            criterion = criterion.cuda()
        
        loss = criterion(logit, target.long())

        return loss
        
if __name__ == "__main__":
    logit = torch.randn((2, 4, 29))
    target = torch.randn((2, 4, 29))
    
    loss_object = loss_collection(False)

    loss_cal = loss_object.construct_loss()

    print(loss_cal(logit, target))