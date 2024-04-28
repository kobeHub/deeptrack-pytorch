import torch
import torch.nn as nn


class WeightedBCELoss(nn.Module):
    def __init__(self):
        super(WeightedBCELoss, self).__init__()
        self.eps = 1e-12

    def forward(self, input, targets):
        target, weights = targets[:, :, 0], targets[:, :, 1]
        target = target.flatten()
        weights = weights.flatten()
        input = input.flatten()
        buffer = torch.empty_like(input)
        buffer = torch.add(input, self.eps).log().mul(weights)
        output = - torch.dot(target, buffer)
        buffer = torch.mul(
            input, -1).add(1).add(self.eps).log().mul(weights)
        output = (output - torch.sum(buffer) +
                  torch.dot(target.view(-1), buffer.view(-1))) / input.numel()
        return output
