import torch
import torch.nn as nn

from Components import *

class VectorNet(nn.Module):
    def __init__(self, depth_sub, width_sub, depth_global, width_global):
        super(VectorNet, self).__init__()
        self.depth_sub = depth_sub
        self.width_sub = width_sub
        self.depth_global = depth_global
        self.width_global = width_global
        self.polyline_embedding = polyline_encoder(depth_sub, width_sub)
        self.global_interaction = global_graph(depth_global, width_global, width_sub)
        self.traj_decode = nn.Linear(width_global, 4)

    def map_encode(self, lane_polylines_batches):
        # map polyline node feature is fixed.
        # shape: [[[4(coordinates), lane_polyline_vector_num], ...], ...]
        self.map_pres = []
        for lane_polylines in lane_polylines_batches:
            # different batch has different count of polylines
            lane_polyline_node_features = []
            for lane_polyline in lane_polylines:
                # different polyline has different count of vectors
                temp = self.polyline_embedding(lane_polyline)
                lane_polyline_node_features.append(temp)

            self.map_pres.append(torch.stack(lane_polyline_node_features, -1))

    def forward(self, traj_batches):
        traj_predict = []
        for traj, map_pre in zip(traj_batches, self.map_pres):
            temp = self.polyline_embedding(traj)
            temp = torch.unsqueeze(temp, -1)
            temp = torch.cat([map_pre, temp], axis=-1)
            temp = self.global_interaction(temp)
            temp = temp[:, :, -1]
            temp = self.traj_decode(temp)
            traj_predict.append(temp)
        
        return torch.cat(traj_predict, axis=0)


if __name__ == '__main__':
    model = VectorNet(3, 64, 1, 128).cuda()

    import random

    a = []
    for i in range(70):
        a.append(torch.randn(1, 4, random.randint(2, 70)).cuda())

    b = []
    for i in range(54):
        b.append(torch.randn(1, 4, random.randint(2, 70)).cuda())

    model.map_encode([a, b]*10)

    x = [torch.randn(1, 4, 32).cuda(), torch.randn(1, 4, 47).cuda()]

    print(model(x).shape)
    torch.save(model.state_dict(), '1.pth')