import torch
import torch.nn as nn


class NCL_linear(nn.Module):
    # THe linear layer of pytorch can only process NWHC or NLC feature map.
    def __init__(self, in_channel, out_channel):
        super(NCL_linear, self).__init__()
        self.mlp = nn.Linear(in_channel, out_channel)
        self.norm = nn.BatchNorm1d(out_channel)
        self.ReLU = nn.ReLU()

    def forward(self, x):
        # NCL -> NLC
        x = x.permute(0, 2, 1)

        x = self.mlp(x)

        # NLC -> NCL
        x = x.permute(0, 2, 1)
        x = self.norm(x)
        x = self.ReLU(x)

        return x


class encode(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(encode, self).__init__()
        self.flatten = lambda x: torch.flatten(x, 1, -2)
        self.mlp = NCL_linear(in_channel, out_channel)

    def forward(self, vectors):
        # vectors: [batch_size, vector_num, 4 or 128]
        if(vectors.dim() > 3):
            vectors = self.flatten(vectors)

        vectors = self.mlp(vectors)

        return vectors


class polyline_subgraph(nn.Module):
    def __init__(self, depth, width):
        super(polyline_subgraph, self).__init__()
        self.depth = depth
        self.width = width
        self.encode = []
        for i in range(depth):
            if(i == 0):
                self.encode.append(encode(4, width))
            else:
                self.encode.append(encode(2*width, width))
        self.maxpool = nn.MaxPool1d

    def forward(self, x):
        for i in range(self.depth):
            x = self.encode[i](x)
        
            x_len = x.shape[-1]
            temp = self.maxpool(x_len, x_len)(x)
            temp = temp.repeat(1, 1, x_len)
            # The channel dimension is 1.
            x = torch.cat([x, temp], axis=1)
        
        x_len = x.shape[-1]
        x = self.maxpool(x_len, x_len)(x)
        x = torch.squeeze(x, -1)

        return x


class global_graph(nn.Module):
    def __init__(self):
        pass


if __name__ == "__main__":
    model = polyline_subgraph(2, 128)

    a = torch.randn([1, 4])
    a = torch.unsqueeze(a, -1)
    a = a.repeat(1, 1, 20)

    out = model(a)
    print(torch.sum(out))
    print(out.shape)