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


class NCHW_linear(nn.Module):
    # THe linear layer of pytorch can only process NWHC or NLC feature map.
    def __init__(self, in_channel, out_channel):
        super(NCHW_linear, self).__init__()
        self.mlp = nn.Linear(in_channel, out_channel)
        self.norm = nn.BatchNorm2d(out_channel)
        self.ReLU = nn.ReLU()
        self.out_channel = out_channel

    def forward(self, x):
        # NCHW -> NHWC
        x = x.permute(0, 2, 3, 1)

        # NHWC -> HLC (L = H*W)
        recov_shape = tuple(x.shape[1:3])
        x = torch.flatten(x, 1, 2)
        x = self.mlp(x)

        # NLC -> NHWC
        x = torch.reshape(x, (-1, *recov_shape, self.out_channel))

        # NHWC -> NCHW
        x = x.permute(0, 3, 1, 2)
        x = self.norm(x)
        x = self.ReLU(x)

        return x


class padding_encode(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(padding_encode, self).__init__()
        self.flatten = lambda x: torch.flatten(x, 1, -3)
        self.mlp = NCHW_linear(in_channel, out_channel)

    def forward(self, vectors):
        # vectors: [batch_size, 4 or polyline_graph_width, vector_num]
        if(vectors.dim() > 4):
            vectors = self.flatten(vectors)

        vectors = self.mlp(vectors)

        return vectors


class padding_polyline_encoder(nn.Module):
    def __init__(self, depth, width):
        super(padding_polyline_encoder, self).__init__()
        self.depth = depth
        self.width = width
        self.encode = []
        for i in range(depth):
            if(i == 0):
                temp = padding_encode(4, width);self.add_linear_layer(temp)
                self.encode.append(temp)
            else:
                temp = padding_encode(2*width, width);self.add_linear_layer(temp)
                self.encode.append(temp)
        self.maxpool = nn.MaxPool2d

    def add_linear_layer(self, x):
        # linear layer need to be put into list but the items in list are not belong to this object.
        setattr(self, "{}".format(hash(x)), x)

    def forward(self, x):
        for i in range(self.depth):
            x = self.encode[i](x)
        
            shape = (x.shape[-2], 1)
            temp = self.maxpool(shape, shape)(x)
            temp = temp.repeat(1, 1, *shape)
            # The channel dimension is 1.
            x = torch.cat([x, temp], axis=1)
        
        shape = (x.shape[-2], 1)
        x = self.maxpool(shape, shape)(x)
        x = torch.squeeze(x, -2)

        return x


class padding_global_graph(nn.Module):
    def __init__(self, depth, width, width_sub):
        super(padding_global_graph, self).__init__()
        self.depth = depth
        self.width = width
        self.width = width_sub
        self.linear_Q = []
        self.linear_K = []
        self.linear_V = []

        for i in range(depth):
            if(i == 0):
                temp_Q = NCL_linear(2*width_sub, width);self.add_linear_layer(temp_Q)
                temp_K = NCL_linear(2*width_sub, width);self.add_linear_layer(temp_K)
                temp_V = NCL_linear(2*width_sub, width);self.add_linear_layer(temp_V)
                self.linear_Q.append(temp_Q)
                self.linear_K.append(temp_K)
                self.linear_V.append(temp_V)
            else:
                temp_Q = NCL_linear(width, width);self.add_linear_layer(temp_Q)
                temp_K = NCL_linear(width, width);self.add_linear_layer(temp_K)
                temp_V = NCL_linear(width, width);self.add_linear_layer(temp_V)
                self.linear_Q.append(temp_Q)
                self.linear_K.append(temp_K)
                self.linear_V.append(temp_V)

        self.softmax = nn.Softmax()

    def add_linear_layer(self, x):
        # linear layer need to be put into list but the items in list are not belong to this object.
        setattr(self, "{}".format(hash(x)), x)

    def forward(self, P_matrix):
        # NCL format
        # The trajectory polyline node feature is one row of the P_matrix.
        for i in range(self.depth):
            PQ = P_matrix; PK = P_matrix; PV = P_matrix
            PQ = self.linear_Q[i](PQ)
            PK = self.linear_K[i](PK)
            PV = self.linear_V[i](PV)
        
            weight = torch.bmm(PQ, torch.transpose(PK, -2, -1))
            out = torch.bmm(self.softmax(weight), PV)
            P_matrix = out

        return P_matrix
        

if __name__ == "__main__":
    model = nn.Sequential(padding_polyline_encoder(3, 64), padding_global_graph(1, 128, 64))

    a = torch.randn(2, 4, 9, 160)

    out = model(a)

    print(out.shape)