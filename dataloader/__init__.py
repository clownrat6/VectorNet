from . import argov_dataset

def construct_loader(train_path, valid_path, batch_size, dataset_name):
    if dataset_name == 'Argoverse':
        train_loader = argov_dataset.dataloader(batch_size, train_path)
        valid_loader = argov_dataset.dataloader(batch_size, valid_path)

        return train_loader, valid_loader
    else:
        assert False, "[Error]:There is no usable dataset loader."
