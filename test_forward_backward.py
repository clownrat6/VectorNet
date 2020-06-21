import warnings
warnings.filterwarnings("ignore")

from utils.losses import *
from utils.process import *
from utils.metricer import *
from modeling.VectorNet import *
from modeling.padding_VectorNet import *
from dataloader.argov_dataset import dataloader

mode = 'loglike'
ifcuda = True

def padding_model_test(model, padding_data_generator):
    """
    One-shot forward inference of padding model.
    """
    loss_object = loss_collection(False)
    loss_cal = loss_object.construct_loss(mode)
    metric = metricer()
    print('padding model (requiring padding data input):')
    count = 0
    for map_pres, train_traj, test_traj in padding_data_generator:
        predict_list = []
        out = model(train_traj, map_pres)
        loss = loss_cal(out, test_traj)
        metric.add_batch(out, test_traj)
        count += 1
        print('batch [{}] predict shape: {} ground truth shape: {} loss: {}'.format(count, out.shape, test_traj.shape, loss))
    DE1s, DE2s, DE3s, ADE = metric.calculate()
    print('metric -- DE@1s:{} DE@2s:{} DE@3s:{} ADE:{}'.format(DE1s, DE2s, DE3s, ADE))
    

def model_test(model, data_generator):
    """
    One-shot forward inference of plain model.
    """
    loss_object = loss_collection(False)
    loss_cal = loss_object.construct_loss(mode)
    metric = metricer()
    print('plain model:')
    count = 0
    for map_pres, train_traj, test_traj in data_generator:
        predict_list = []
        out = model(train_traj, map_pres)
        # print(train_traj[0].shape, len(train_traj))
        # print(out.shape)
        loss = loss_cal(out, test_traj)
        metric.add_batch(out, test_traj)
        count += 1
        print('batch [{}] predict shape: {} ground truth shape: {} loss: {}'.format(count, out.shape, test_traj.shape, loss))
    DE1s, DE2s, DE3s, ADE = metric.calculate()
    print('metric -- DE@1s:{} DE@2s:{} DE@3s:{} ADE:{}'.format(DE1s, DE2s, DE3s, ADE))

batch_size = 2

dataset = dataloader(batch_size, ifcuda)

padding_data_generator = dataset.padding_dataset_generator()
data_generator = dataset.dataset_generator()

model_1 = padding_VectorNet(3, 64, 1, 128)
model_2 = VectorNet(3, 64, 1, 128)

if ifcuda:
    model_1 = model_1.cuda()
    model_2 = model_2.cuda()

padding_model_test(model_1, padding_data_generator)
model_test(model_2, data_generator)
