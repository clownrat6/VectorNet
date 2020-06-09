import torch
import torch.nn as nn

from .padding_Components import *

class padding_VectorNet(nn.Module):
    """
    A padding implementation of VectorNet which is suitable to process scenario thai has same count of polyline.
    """
    def __init__(self, depth_sub, width_sub, depth_global, width_global):
        super(padding_VectorNet, self).__init__()
        self.depth_sub = depth_sub
        self.width_sub = width_sub
        self.depth_global = depth_global
        self.width_global = width_global
        self.polyline_embedding = padding_polyline_encoder(depth_sub, width_sub)
        self.global_interaction = padding_global_graph(depth_global, width_global, width_sub)
        self.traj_decode = nn.Linear(width_global, 2*30)

    def forward(self, traj_batches, lane_polylines_batches):
        map_pres = self.polyline_embedding(lane_polylines_batches)
        traj_batches = self.polyline_embedding(torch.unsqueeze(traj_batches, -1))
        temp = torch.cat([map_pres, traj_batches], axis=-1)
        temp = self.global_interaction(temp)
        temp = temp[:, :, -1]
        temp = self.traj_decode(temp)
        length = temp.shape[-1]
        temp = temp.reshape((-1, 2, int(length/2)))

        return temp
        

if __name__ == '__main__':
    model = padding_VectorNet(3, 64, 1, 128)

    map_vectors = torch.randn(2, 4, 9, 160)

    model.map_encode(map_vectors)

    trajectory = torch.randn(2, 4, 19)
    
    out = model(trajectory)
    
    print(out.shape)
    torch.save(model.state_dict(), 'test.pth')