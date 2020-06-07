import torch
import torch.nn

import numpy as np

class metricer(object):
    def __init__(self):
        self.second_1_pr = []; self.second_1_gt = []
        self.second_2_pr = []; self.second_2_gt = []
        self.second_3_pr = []; self.second_3_gt = []

    def add_batch(self, pr, gt):
        self.second_1_pr.append(pr[:, :, 9]); self.second_1_gt.append(gt[:, :, 9])
        self.second_2_pr.append(pr[:, :, 19]); self.second_2_gt.append(gt[:, :, 19])
        self.second_3_pr.append(pr[:, :, 29]); self.second_3_gt.append(gt[:, :, 29])

    def calculate(self):
        self.second_1_pr = torch.cat(self.second_1_pr, axis=0)
        self.second_1_gt = torch.cat(self.second_1_gt, axis=0)
        self.second_2_pr = torch.cat(self.second_2_pr, axis=0)
        self.second_2_gt = torch.cat(self.second_2_gt, axis=0)
        self.second_3_pr = torch.cat(self.second_3_pr, axis=0)
        self.second_3_gt = torch.cat(self.second_3_gt, axis=0)

        DE1s = torch.mean(torch.abs(self.second_1_pr - self.second_1_gt))
        DE2s = torch.mean(torch.abs(self.second_2_pr - self.second_2_gt))
        DE3s = torch.mean(torch.abs(self.second_3_pr - self.second_3_gt))
        ADE = (DE1s + DE2s + DE3s)/3

        return DE1s, DE2s, DE3s, ADE

    def flush(self):
        self.second_1_pr = []; self.second_1_gt = []
        self.second_2_pr = []; self.second_2_gt = []
        self.second_3_pr = []; self.second_3_gt = []