import warnings
warnings.filterwarnings("ignore")

from utils.losses import *
from utils.process import *
from utils.metricer import *
from modeling.VectorNet import *
from modeling.padding_VectorNet import *
from dataloader.argov_dataset import dataloader

mode='loglike'

def padding_model_test(model, padding_data_generator):
    loss_object = loss_collection(False)
    loss_cal = loss_object.construct_loss(mode)
    metric = metricer()
    print('padding model (requiring padding data input):')
    count = 0
    for map_pres, train_traj, test_traj in padding_data_generator:
        predict_list = []
        model.map_encode(map_pres)
        out = model(train_traj)
        loss = loss_cal(out, test_traj)
        metric.add_batch(out, test_traj)
        count += 1
        print('batch [{}] predict shape: {} ground truth shape: {} loss: {}'.format(count, out.shape, test_traj.shape, loss))
    DE1S, DE2s, DE3s, ADE = metric.calculate()
    print('metric -- DE@1s:{} DE@2s:{} DE@3s:{} ADE:{}'.format(DE1S, DE2s, DE3s, ADE))
    

def model_test(model, data_generator):
    loss_object = loss_collection(False)
    loss_cal = loss_object.construct_loss(mode)
    metric = metricer()
    print('plain model:')
    count = 0
    for map_pres, train_traj, test_traj in data_generator:
        predict_list = []
        model.map_encode(map_pres)
        out = model(train_traj)
        loss = loss_cal(out, test_traj)
        metric.add_batch(out, test_traj)
        count += 1
        print('batch [{}] predict shape: {} ground truth shape: {} loss: {}'.format(count, out.shape, test_traj.shape, loss))
    DE1S, DE2s, DE3s, ADE = metric.calculate()
    print('metric -- DE@1s:{} DE@2s:{} DE@3s:{} ADE:{}'.format(DE1S, DE2s, DE3s, ADE))

batch_size = 2

dataset = dataloader(batch_size)

padding_data_generator = dataset.padding_dataset_generator()
data_generator = dataset.dataset_generator()

model = padding_VectorNet(3, 64, 1, 128)
padding_model_test(model, padding_data_generator)

model = VectorNet(3, 64, 1, 128)
model_test(model, data_generator)
