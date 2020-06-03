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


class polyline_encoder(nn.Module):
    def __init__(self, depth, width):
        super(polyline_encoder, self).__init__()
        self.depth = depth
        self.width = width
        self.encode = []
        for i in range(depth):
            if(i == 0):
                temp = encode(4, width);self.add_linear_layer(temp)
                self.encode.append(temp)
            else:
                temp = encode(2*width, width);self.add_linear_layer(temp)
                self.encode.append(temp)

        self.maxpool = nn.MaxPool1d

    def add_linear_layer(self, x):
        setattr(self, "{}".format(hash(x)), x)

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
    def __init__(self, depth, width, width_sub):
        super(global_graph, self).__init__()
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
    model = global_graph(3, 64, 128)

    torch.save(model.state_dict(), '1.pth')

    exit()
    a = torch.randn(2, 128, 99)

    out = model(a)
    print(torch.sum(out))
    print(out.shape)