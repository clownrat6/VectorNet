import torch
import torch.nn as nn

from padding_Components import *

class VectorNet(nn.Module):
    def __init__(self, depth_sub, width_sub, depth_global, width_global):
        super(VectorNet, self).__init__()
        self.depth_sub = depth_sub
        self.width_sub = width_sub
        self.depth_global = depth_global
        self.width_global = width_global
        self.polyline_embedding = padding_polyline_encoder(depth_sub, width_sub)
        self.global_interaction = padding_global_graph(depth_global, width_global, width_sub)
        self.traj_decode = nn.Linear(width_global, 4)

    def map_encode(self, lane_polylines_batches):
        # map polyline node feature is fixed.
        # shape: [[[4(coordinates), lane_polyline_vector_num], ...], ...]
        self.map_pres = self.polyline_embedding(lane_polylines_batches)
        return self.map_pres

    def forward(self, traj_batches):
        traj_batches = self.polyline_embedding(traj_batches)
        temp = torch.cat([self.map_pres, traj_batches], axis=-1)
        temp = self.global_interaction(temp)
        temp = temp[:, :, -1]
        temp = self.traj_decode(temp)

        return temp
        

if __name__ == '__main__':
    model = VectorNet(3, 64, 1, 128)

    import random

    a = torch.randn(2, 4, 37, 49)

    print(model.map_encode(a).shape)
    b = torch.randn(2, 4, 31, 1)
    print(model(b).shape)
    exit()
    torch.save(model.state_dict(), '1.pth')