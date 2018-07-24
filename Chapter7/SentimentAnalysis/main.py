import torch
from corpus import corpus
import numpy as np
from model import BILSTM
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import sys
from tqdm import tqdm
import os
from visdom import Visdom
import time


batch_size=64
epochs_size=100
hidden_size=128
cuda=torch.cuda.is_available()
num_layers=2

def train(corpus,model):
    viz = Visdom()
    line = viz.line(np.arange(2))

    step_p=[]
    train_loss_p=[]
    dev_loss_p=[]
    train_acc_p=[]
    dev_acc_p=[]


    optimizer = torch.optim.Adam(model.parameters(), lr=0.0002)
    criterion = nn.CrossEntropyLoss()
    for epoch in range(epochs_size):
        train_loss=[]
        train_acc=[]
        for idx in tqdm(range(0,len(corpus.train_ids),batch_size)):
            states = (Variable(torch.zeros(num_layers*2,batch_size,hidden_size)).cuda(),
                      Variable(torch.zeros(num_layers*2,batch_size,hidden_size)).cuda())
            if cuda:
                inp=Variable(torch.from_numpy(corpus.train_ids[idx:idx+64])).cuda()
                tag=Variable(torch.from_numpy(corpus.train_tags[idx:idx+64])).cuda()
            if inp.size(0)!=batch_size:
                continue
            pred=model(inp,states)
            _,pred_idx=torch.max(pred,1)
            loss = criterion(pred,tag)
            optimizer.zero_grad()
            loss.backward()
            # 剪裁参数梯度
            nn.utils.clip_grad_norm(model.parameters(), 1, norm_type=2)
            optimizer.step()
            train_loss.append(loss.data[0])
            train_acc.append((sum(pred_idx.cpu().data.numpy()==tag.cpu().data.numpy())*1./tag.size(0)))


        dev_loss=[]
        dev_acc=[]
        # 查看测试集损失
        for idx in tqdm(range(0,len(corpus.eval_ids),batch_size)):
            states = (Variable(torch.zeros(num_layers*2,batch_size,hidden_size)).cuda(),
                      Variable(torch.zeros(num_layers*2,batch_size,hidden_size)).cuda())
            if cuda:
                inp=Variable(torch.from_numpy(corpus.train_ids[idx:idx+64])).cuda()
                tag=Variable(torch.from_numpy(corpus.train_tags[idx:idx+64])).cuda()
            if inp.size(0)!=batch_size:
                continue
            pred=model(inp,states)
            _,pred_idx=torch.max(pred,1)
            loss = criterion(pred,tag)
            # 剪裁参数梯度
            dev_loss.append(loss.data[0])
            dev_acc.append((sum(pred_idx.cpu().data.numpy()==tag.cpu().data.numpy())*1./tag.size(0)))

        print("epoch :{},train mean loss:{},dev mean loss:{}".format(epoch,np.mean(train_loss),np.mean(dev_loss)))
        train_loss_p.append(np.mean(train_loss))
        dev_loss_p.append(np.mean(dev_loss))
        train_acc_p.append(np.mean(train_acc))
        dev_acc_p.append(np.mean(dev_acc))
        step_p.append(epoch)
        viz.line(
             X=np.column_stack((np.array(step_p), np.array(step_p),np.array(step_p), np.array(step_p))),
             Y=np.column_stack((np.array(train_loss_p),np.array(train_acc_p),np.array(dev_loss_p), np.array(dev_acc_p))),
             win=line,
            opts=dict(legend=["Train_mean_loss", "Train_acc","Eval_mean_loss", "Eval_acc"]))


if __name__ == '__main__':
    corpus=corpus("data/pos.csv","data/neg.csv","data/neutral.csv","data/stop_words.csv")
    model=BILSTM(corpus.num_classes,corpus.vocab_size,hidden_size,num_layers)
    if cuda:
        model=model.cuda()
    train(corpus,model)
