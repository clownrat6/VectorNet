import torch
import torch.nn as nn

from .Components import *

class VectorNet(nn.Module):
    """
    This is one type implementation of VectorNet.
    This implementation is suitable to unpadding lane polyline.
    """
    def __init__(self, depth_sub, width_sub, depth_global, width_global):
        super(VectorNet, self).__init__()
        self.depth_sub = depth_sub
        self.width_sub = width_sub
        self.depth_global = depth_global
        self.width_global = width_global
        self.polyline_embedding = polyline_encoder(depth_sub, width_sub)
        self.global_interaction = global_graph(depth_global, width_global, width_sub)
        self.traj_decode = nn.Linear(width_global, 2*30)

    def map_encode(self, lane_polylines_batches):
        """
        Map representation like lane two-side line is fixed. So we use a object attribute to keep it.
        input:
            lane_polylines_batches: [lane_polylines]*batch_size. the number of polylines are different between items in batch.
                p.s. lane_polylines = [lane_polyline]*polyline_num, lane_polyline.shape = [4(coordinates, lane_polyline_vector_num)]  
        """
        # map polyline node feature is fixed.
        # shape: [[[4(coordinates), lane_polyline_vector_num], ...], ...]
        self.map_pres_batch = []
        for lane_polylines in lane_polylines_batches:
            # different batch has different count of polylines
            lane_polyline_node_features = []
            for lane_polyline in lane_polylines:
                # different polyline has different count of vectors
                lane_polyline = torch.unsqueeze(lane_polyline, 0)
                temp = self.polyline_embedding(lane_polyline)
                lane_polyline_node_features.append(temp)

            self.map_pres_batch.append(torch.stack(lane_polyline_node_features, -1))

    def forward(self, traj_batch):
        traj_predict_batch = []
        for traj, map_pres in zip(traj_batch, self.map_pres_batch):
            traj = torch.unsqueeze(traj, 0)
            temp = self.polyline_embedding(traj)
            temp = torch.unsqueeze(temp, -1)
            temp = torch.cat([map_pres, temp], axis=-1)
            temp = self.global_interaction(temp)
            temp = temp[:, :, -1]
            traj_predict = self.traj_decode(temp)
            length = traj_predict.shape[-1]
            traj_predict = traj_predict.reshape((-1, 2, int(length/2)))
            traj_predict_batch.append(traj_predict)
        
        return torch.cat(traj_predict_batch, 0)


if __name__ == '__main__':
    model = VectorNet(3, 64, 1, 128).cuda()

    import random

    map1 = []
    for i in range(70):
        # 70 polyline
        map1.append(torch.randn(4, random.randint(2, 70)).cuda())

    map2 = []
    for i in range(54):
        # 54 polyline
        map2.append(torch.randn(4, random.randint(2, 70)).cuda())

    map_batch = [map1, map2]

    model.map_encode(map_batch)

    traj_batch = [torch.randn(4, 19).cuda(), torch.randn(4, 19).cuda()]

    out = model(traj_batch)

    print(out.shape)

    torch.save(model.state_dict(), 'test.pth')