import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import sys

class MLP(nn.Module):
    """A MLP for text classification

    architecture: Embedding >> Convolution >> Max-pooling >> Softmax
    """

    def __init__(self, num_classes, vocab_size, emb_dim):
        super(MLP, self).__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim)
        self.lin1 = nn.Linear(62336, 1024)
        self.lin2 = nn.Linear(1024, num_classes)
        self.init_parameters()


    def forward(self, x):
        emb = self.emb(x)  # 32 * 487 * 128
        emb=emb.view(x.size(0),-1)
        out=self.lin1(emb)
        out=self.lin2(out)
        out=F.softmax(out, dim=1)
        return out


    def init_parameters(self):
        for param in self.parameters():
            param.data.uniform_(-0.05, 0.05)
