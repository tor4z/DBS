import torch
from torchvision import transforms
import copy
import numpy as np
from torch.utils.data import Dataset
import pandas as pd


class DataContainer(object):
    def __init__(self):
        self.data = []
        self.len = -1

    def init(self):
        self.len = len(self.data)
        self.shuffled_index = list(range(self.len))
        np.random.shuffle(self.shuffled_index)

    def __len__(self):
        return self.len

    def append(self, data):
        self.data.append(data)

    def __getitem__(self, index):
        if index == 0:
            np.random.shuffle(self.shuffled_index)
        index = index % self.len
        index = self.shuffled_index[index]
        return self.data[index]


class DigitDataset(Dataset):
    def __init__(self, opt, data, test=False, validate=False):
        self.test = test
        self.validate = validate
        self.eye = torch.eye(10)
        self.datas = []
        if self.validate or self.test:
            self.data = data
            self.data_len = len(data)
        else:
            self.flatten(data)

    def flatten(self, data):
        self.data = [DataContainer() for _ in range(10)]
        data_len = len(data)
        for i in range(data_len):
            label = data[i, 0]
            image = data[i, 1:]
            self.data[label].append(image)

        max_len = 0
        for i, lst in enumerate(self.data):
            lst.init()
            lst_len = len(lst)
            print(f'Label {i} length {lst_len}')
            if max_len < lst_len:
                max_len = lst_len
        print(f'Max length: {max_len}')
        self.data_len = max_len * 10

    def __len__(self):
        return self.data_len

    def one_hot(self, label):
        if label > 9 or label < 0:
            raise ValueError(f'({label}) label out of range.')
        return self.eye[label]

    def get_item(self, index):
        if not (self.test or self.validate):
            label = (index % 10)
            image = self.data[label][index // 10]
        elif self.test:
            label = -1                   # to be predict
            image = self.data[index, :]
        else:
            label = self.data[index, 0]
            image = self.data[index, 1:]
        
        return image, label

    def __getitem__(self, index):
        image, label = self.get_item(index)
        image = torch.tensor(image).type(torch.float32).view(28, 28) / 255.0
        image = image.unsqueeze(0)
        label = self.one_hot(label)

        return index, image, label


def get_train_val_test_dataset(opt):
    all_train_data = data_reader(opt.train_path)
    test_data = data_reader(opt.test_path)
    
    train_len = len(all_train_data)

    ids = [i for i in range(train_len)]
    np.random.shuffle(ids)

    train_data_len = int(train_len * opt.train_rate)
    validate_data_len = train_len - train_data_len

    validate_ids = copy.copy(ids[: validate_data_len])
    del ids[:validate_data_len]
    train_ids = ids

    train_data = all_train_data[train_ids, :]
    validate_data = all_train_data[validate_ids, :]

    train_dataset = DigitDataset(opt, train_data)
    validate_dataset = DigitDataset(opt, validate_data, validate=True)
    test_dataset = DigitDataset(opt, test_data, test=True)

    return train_dataset, validate_dataset, test_dataset


def data_reader(path):
    df = pd.read_csv(path)
    return df.to_numpy()


if __name__ == "__main__":
    data = data_reader('.data/train.csv')
    print(data)