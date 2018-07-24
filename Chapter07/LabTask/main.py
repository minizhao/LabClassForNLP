from data_loader import corpus
from  tqdm import tqdm
from  model  import MLP
import sys
import torch
import numpy as np
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import sys
from tqdm import tqdm
import os
from visdom import Visdom
import time

batch_size=32

def train(c,mlp,optimizer,criterion):
    for _ in range(20):
        for idx in tqdm(range(0,len(c.train_ids),batch_size)):
            inp=Variable(torch.from_numpy(c.train_ids[idx:idx+batch_size]))
            tag=Variable(torch.from_numpy(c.train_tags[idx:idx+batch_size]))
            pred=mlp(inp)
            loss = criterion(pred,tag)
            print(loss)

            optimizer.zero_grad()
            loss.backward()
                # 剪裁参数梯度
            nn.utils.clip_grad_norm(mlp.parameters(), 1, norm_type=2)
            optimizer.step()
    # print('eval_mean_loss:{},eval_mean_acc:{}'.format(np.mean(loss_list),np.mean(acc_list)))


if __name__ == '__main__':
    c=corpus()
    criterion = nn.CrossEntropyLoss()
    mlp=MLP(num_classes=c.num_class,vocab_size=c.vocab_size, emb_dim=128)
    optimizer=optim.Adam(mlp.parameters(),lr=0.0005)
    train(c,mlp,optimizer,criterion)
