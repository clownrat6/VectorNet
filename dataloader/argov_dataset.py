import sys
sys.path.append('..')

import torch
import random

from utils.process import *

def map_representation_padding(map_pres, map_polyline_num, polyline_vector_num):
    for polyline in map_pres: 
        if len(polyline) >= polyline_vector_num:
            polyline = polyline[:polyline_vector_num]
        else:
            for _ in range(polyline_vector_num - len(polyline)):
                polyline.append(np.zeros(4))

    if len(map_pres) >= map_polyline_num:
        map_pres = map_pres[:map_polyline_num]
    else:
        for _ in range(map_polyline_num - len(map_pres)):
            map_pres.append([np.zeros(4)]*polyline_vector_num)
            
    return map_pres


def form_padding_batch(scenario_list, map_polyline_num, polyline_vector_num):
    train_trajectory_batch = []
    test_trajectory_batch = []
    map_vectors_batch = []
    for scenario in scenario_list:
        map_vectors, train_trajectory, test_trajectory = scenario_vectorization(scenario)
        map_vectors = map_representation_padding(map_vectors, map_polyline_num, polyline_vector_num)
        map_vectors_batch.append(np.array(map_vectors))
        train_trajectory_batch.append(train_trajectory)
        test_trajectory_batch.append(test_trajectory)

    map_vectors_batch = torch.tensor(map_vectors_batch, dtype=torch.float32).permute(0, 3, 2, 1)
    train_trajectory_batch = torch.tensor(train_trajectory_batch, dtype=torch.float32).permute(0, 2, 1)
    test_trajectory_batch = torch.tensor(test_trajectory_batch, dtype=torch.float32).permute(0, 2, 1)

    return map_vectors_batch, train_trajectory_batch, test_trajectory_batch

def form_batch(scenario_list):
    train_trajectory_batch = []
    test_trajectory_batch = []
    map_vectors_batch = []
    for scenario in scenario_list:
        map_vectors, train_trajectory, test_trajectory = scenario_vectorization(scenario)
        map_vectors = [torch.tensor(x, dtype=torch.float32).permute(1, 0) for x in map_vectors]
        map_vectors_batch.append(map_vectors)
        train_trajectory_batch.append(torch.tensor(train_trajectory, dtype=torch.float32).permute(1, 0))
        test_trajectory_batch.append(test_trajectory)

    test_trajectory_batch = torch.tensor(test_trajectory_batch, dtype=torch.float32).permute(0, 2, 1)

    return map_vectors_batch, train_trajectory_batch, test_trajectory_batch


class dataloader(object):
    def __init__(self, batch_size, data_path=None):
        if data_path == None:
            self.ap = argoverse_processor()
        else:
            self.ap = argoverse_processor(data_path)
        
        self.scenarios = [scenario_object(x, self.ap) for x in self.ap.scenarios]
        
        self.batch_size = batch_size

    def padding_dataset_generator(self, map_polyline_num = 160, polyline_vector_num = 9):
        batch_size = self.batch_size
        assert batch_size <= len(self.scenarios), 'batch size is larger than the scale of dataset.'
        random.shuffle(self.scenarios)
        scenario_list = []
        for scenario in self.scenarios:
            scenario_list.append(scenario)
            if len(scenario_list) == batch_size:
                yield form_padding_batch(scenario_list, \
                    map_polyline_num = map_polyline_num, polyline_vector_num = polyline_vector_num)
                scenario_list = []

    def dataset_generator(self):
        batch_size = self.batch_size
        assert batch_size <= len(self.scenarios), 'batch size is larger than the scale of dataset.'
        random.shuffle(self.scenarios)
        scenario_list = []
        for scenario in self.scenarios:
            scenario_list.append(scenario)
            if len(scenario_list) == batch_size:
                yield form_batch(scenario_list)
                scenario_list = []

if __name__ == "__main__":
    dataset = dataloader(2)

    data_generator = dataset.dataset_generator()

    for a,b,c in data_generator:
        print(a[0][0].shape, b[0].shape, c[0].shape)

    data_generator = dataset.padding_dataset_generator()

    for a,b,c in data_generator:
        print(a.shape, b.shape, c.shape)