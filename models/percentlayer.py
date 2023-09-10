import torch
from torch import nn

class PercentLayer(nn.Module):
    def __init__(self, CTG, W, AIR):
        super().__init__()
        self.CTG = CTG
        self.W = W
        self.AIR = AIR
    """3 input. 1 output"""
    def forward(self, x):

        x = torch.softmax(x, 1)

        output = self.CTG * x[:, 0] + self.W * x[:, 1] + self.AIR * x[:, 2]
        output = output.unsqueeze(1)
        rescale = (output - self.AIR) * (1 / (self.CTG - self.AIR))
        rescale = 2 * rescale - 1
        return rescale
