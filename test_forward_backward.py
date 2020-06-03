

from utils.process import *
from modeling.VectorNet import *
from modeling.padding_VectorNet import *
from dataloader.argov_dataset import dataloader

def padding_model_test(model, padding_data_generator):
    for map_pres, train_traj, test_traj in padding_data_generator:
        predict_list = []
        model.map_encode(map_pres)
        for i in range(test_traj.shape[-1]):
            out = model(train_traj)
            predict_list.append(out)
            train_traj = torch.cat([train_traj, torch.unsqueeze(out, -1)], -1)
        out = torch.stack(predict_list, -1)
        print(out.shape, test_traj.shape)
        return None

def model_test(model, data_generator):
    for map_pres, train_traj, test_traj in data_generator:
        predict_list = []
        model.map_encode(map_pres)
        for i in range(test_traj.shape[-1]):
            out = model(train_traj)
            predict_list.append(out)
            train_traj = [torch.cat([train_traj[i], torch.unsqueeze(out[i], -1)], -1) for i in range(len(train_traj))]
        out = torch.stack(predict_list, -1)
        print(out.shape, test_traj.shape)
        return None

batch_size = 2

dataset = dataloader(batch_size)

padding_data_generator = dataset.padding_dataset_generator()
data_generator = dataset.dataset_generator()

model = padding_VectorNet(3, 64, 1, 128)
padding_model_test(model, padding_data_generator)

model = VectorNet(3, 64, 1, 128)
model_test(model, data_generator)
