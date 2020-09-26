#from torch.nn.modules.module import Module
from models.nndistance.functions.nnd import NNDFunction
import torch.nn as nn


class NNDModule(nn.Module):
    def forward(self, input1, input2):
        return NNDFunction()(input1, input2)
