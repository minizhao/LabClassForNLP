import torch
import torch.nn as nn
import sys
import torch.nn.functional as F


class BILSTM(nn.Module):
    """
    一个双向的LSTM实现文本3类别情感分析，类别分别是积极，消极，中立
    """

    def __init__(self, num_classes, vocab_size,hidden_size,num_layers, emb_dim=128, dropout=0.2):
        super(BILSTM, self).__init__()
        self.embed = nn.Embedding(vocab_size, emb_dim)

        self.lstm = nn.LSTM(emb_dim, hidden_size, num_layers, batch_first=True,bidirectional=True)
        self.linear = nn.Linear(1024, num_classes)
        self.init_weights()
        # self.dropout = nn.Dropout(p=dropout)

    def init_weights(self):
        self.embed.weight.data.uniform_(-0.1, 0.1)
        self.linear.bias.data.fill_(0)
        self.linear.weight.data.uniform_(-0.1, 0.1)

    def forward(self, x,h):
        x = self.embed(x)
        # Forward propagate RNN
        _, (h_n,c_n) = self.lstm(x, h)
        features=torch.cat((h_n.view(x.size(0),-1),c_n.view(x.size(0),-1)),1)
        output=self.linear(features)
        output=F.log_softmax(output, dim=1)
        return output
