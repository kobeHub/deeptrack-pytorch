import numpy as np
import torch
from torch.utils.data import Dataset
import torchfile


class SensorData(Dataset):
    def __init__(self, file, params, sequence_length=100, dropout_interval=20):
        self.params = params
        self.sequence_length = sequence_length
        self.dropout_interval = dropout_interval
        # load raw 1D depth sensor data
        data = torchfile.load(file)
        self.data = torch.tensor(data, dtype=torch.float32)
        self.width = int(
            (params['grid_maxX'] - params['grid_minX']) / params['grid_step'])
        self.height = int(
            (params['grid_maxY'] - params['grid_minY']) / params['grid_step'])
        print(f'Width: {self.width}, Height: {self.height}')
        # pre-compute lookup arrays
        self.dist = torch.zeros((self.height, self.width), dtype=torch.float32)
        self.index = torch.zeros((self.height, self.width), dtype=torch.int64)
        for y in range(self.height):
            for x in range(self.width):
                px = (x * params['grid_step'] + params['grid_minX'])
                py = (y * params['grid_step'] + params['grid_minY'])

                angle = np.degrees(np.arctan2(px, py))
                self.dist[y, x] = np.sqrt(px * px + py * py)
                self.index[y, x] = int(
                    (angle - params['sensor_start']) / params['sensor_step'] + 1.5) - 1
        self.index = self.index.reshape(self.width * self.height)

    def __len__(self):
        return len(self.data) // self.sequence_length

    def __getitem__(self, idx):
        """
        Return a tensor of shape (2, height, width) representing the input
        """
        start_idx = idx * self.sequence_length
        inputs = []
        targets = []
        for i in range(self.sequence_length):
            dist = self.data[start_idx + i].index_select(
                0, self.index.clone().detach()).view(self.height, self.width)
            real_data = torch.zeros(2, self.height, self.width)
            real_data[0] = (
                dist - self.dist).abs() < self.params['grid_step'] * 0.7071
            real_data[1] = (dist + self.params['grid_step']
                            * 0.7071) > self.dist
            input = real_data.clone()
            if i % self.dropout_interval >= self.dropout_interval // 2:
                input.zero_()
            inputs.append(input)
            targets.append(real_data)
        return torch.stack(inputs), torch.stack(targets)
        # dist = self.data[i].index_select(0,
        #                                  self.index.clone().detach()).view(self.height, self.width)
        # input = torch.zeros(2, self.height, self.width)
        # input[0] = (dist - self.dist).abs() < self.params['grid_step'] * 0.7071
        # input[1] = (dist + self.params['grid_step'] * 0.7071) > self.dist
        # return input
