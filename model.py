import torch
from torch import nn


class RecurrentUnit(nn.Module):
    def __init__(self, width, height):
        super(RecurrentUnit, self).__init__()
        self.width = width
        self.height = height
        self.conv1 = nn.Conv2d(2, 16, kernel_size=7, stride=1, padding=3)
        self.conv2 = nn.Conv2d(48, 32, kernel_size=7, stride=1, padding=3)
        self.conv3 = nn.Conv2d(32, 1, kernel_size=7, stride=1, padding=3)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x1, h0):
        e = self.sigmoid(self.conv1(x1))
        # h0 = h0.view(h0.shape[0], -1, self.height, self.width)
        j = torch.cat((e, h0), dim=1)
        h1 = self.sigmoid(self.conv2(j))
        y1 = self.sigmoid(self.conv3(h1))
        return h1, y1

    def get_initial_state(self, batch_size=32):
        return torch.randn(batch_size, 32, self.height, self.width)


class DeepTrackingRNN(nn.Module):
    def __init__(self, width, height, num_units):
        super(DeepTrackingRNN, self).__init__()
        self.width = width
        self.height = height
        self.num_units = num_units
        self.step_module = RecurrentUnit(width, height)

    def forward(self, x):
        """
        X shape: B, S, C, H, W
        """
        B, S, C, H, W = x.shape
        x = x.view(-1, C, H, W)
        h = self.step_module.get_initial_state(x.shape[0]).to(x.device)
        outputs = []
        for _ in range(self.num_units):
            h, y = self.step_module(x, h)
            y = y.view(B, S, -1, self.height, self.width)
            outputs.append(y)
        return torch.stack(outputs, dim=1)

    def save(self, path):
        ckpt = {
            'config': {
                'width': self.width,
                'height': self.height,
                'num_units': self.num_units,
            },
            'weight': self.state_dict()
        }
        torch.save(ckpt, path)

    @classmethod
    def from_pretrained(cls, path):
        ckpt = torch.load(path)
        config = ckpt['config']
        model = cls(config['width'], config['height'], config['num_units'])
        model.load_state_dict(ckpt['weight'])
        return model
