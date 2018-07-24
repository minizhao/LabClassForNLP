import torch
from corpus import corpus
import numpy as np
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import sys
from tqdm import tqdm
import os
from visdom import Visdom
import time
import torch.nn.functional as F



class Encoder(nn.Module):
    """
    一个双向的LSTM实现文本编码的作用
    """

    def __init__(self,hidden_size,num_layers=2, emb_dim=128, dropout=0.2):
        super(Encoder, self).__init__()
        self.num_layers=num_layers
        self.hidden_size=hidden_size
        self.lstm = nn.LSTM(emb_dim, hidden_size, num_layers, batch_first=True,bidirectional=True)
        self.linear = nn.Linear(hidden_size*8, hidden_size)
        self.init_weights()
        self.dropout = nn.Dropout(p=dropout)

    def init_weights(self):
        self.linear.bias.data.fill_(0)
        self.linear.weight.data.uniform_(-0.1, 0.1)

    def get_states(self,batch_size):
        states = (Variable(torch.zeros(self.num_layers*2,batch_size,self.hidden_size)).cuda(),
                  Variable(torch.zeros(self.num_layers*2,batch_size,self.hidden_size)).cuda())
        return states

    def forward(self, x):
        h=self.get_states(x.size(0))
        # Forward propagate RNN
        _, (h_n,c_n) = self.lstm(x, h)
        features=torch.cat((h_n.view(x.size(0),-1),c_n.view(x.size(0),-1)),1)
        output=self.linear(features)
        return output


class Decoder(nn.Module):
    """
    解码器生成对话
    """

    def __init__(self,vocab_size,hidden_size,num_layers=2, emb_dim=128, dropout=0.2):
        super(Decoder, self).__init__()

        self.lstm = nn.LSTM(emb_dim, hidden_size, num_layers=2,batch_first=True,bidirectional=False)
        self.linear = nn.Linear(hidden_size, vocab_size)
        self.proj= nn.Linear(hidden_size+emb_dim, emb_dim)
        self.init_weights()
        self.dropout = nn.Dropout(p=dropout)
        self.vocab_size=vocab_size

    def init_weights(self):
        self.linear.bias.data.fill_(0)
        self.linear.weight.data.uniform_(-0.1, 0.1)
        self.proj.bias.data.fill_(0)
        self.proj.weight.data.uniform_(-0.1, 0.1)

    def get_states(self,enc_output):
        states = (enc_output.unsqueeze(0).repeat(2,1,1),
                  enc_output.unsqueeze(0).repeat(2,1,1))
        return states

    def forward(self,enc_output,target,corpus,embed,train=True):
        h=self.get_states(enc_output)
        if train:
            dec_output=Variable(torch.zeros(target.size(0),target.size(1),self.vocab_size)).cuda()
            for i in range(target.size(1)):
                inp=self.proj(torch.cat((target[:,i,:],enc_output),1))
                out,h=self.lstm(inp.unsqueeze(1),h)
                out=self.linear(out.squeeze(1))
                dec_output[:,i,:]=out

            dec_output=dec_output.view(-1,self.vocab_size)
            dec_output=F.log_softmax(dec_output, dim=1)
            return dec_output
        else:
            resulst=[]
            dec_output=[]
            h=self.get_states(enc_output)
            inp=Variable(torch.from_numpy(np.array([corpus.token2idx["<s>"]]*enc_output.size(0)))).cuda()
            for i in range(50):
                inp=embed(inp)
                inp=self.proj(torch.cat((inp,enc_output),1))
                out,h=self.lstm(inp.unsqueeze(1),h)
                out=self.linear(out.squeeze(1))
                out=F.softmax(out, dim=1)
                out=torch.multinomial(out,1).squeeze(1).contiguous()
                dec_output.append(out.cpu().numpy())
                inp=out
            rst=(np.array(dec_output).T)
            for r in rst:
                ans="".join([corpus.idx2token[x] for x in r if x!=corpus.token2idx["<s>"] and x!=corpus.token2idx["<e>"] and x!=corpus.token2idx["<pad>"]])
                resulst.append(ans)
            return resulst


class Seq2Seq(nn.Module):
    """
    Seq2Seq模型生成对话系统
    """

    def __init__(self,vocab_size,hidden_size=256, emb_dim=128, dropout=0.2):
        super(Seq2Seq, self).__init__()
        self.embed = nn.Embedding(vocab_size, emb_dim)
        self.dropout = nn.Dropout(p=dropout)
        self.encoder=Encoder(hidden_size)
        self.decoder=Decoder(vocab_size,hidden_size)
        self.init_weights()

    def init_weights(self):
        self.embed.weight.data.uniform_(-0.1, 0.1)

    def forward(self,question,answer,corpus,train=True):
        question = self.embed(question)
        if train:
            answer = self.embed(answer)
        # Forward propagate RNN
        enc_output=self.encoder(question)
        dec_output=self.decoder(enc_output,answer,corpus,self.embed,train)

        return dec_output
