import torch
from corpus import corpus
import numpy as np
from text_cnn import CNN
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import sys
from tqdm import tqdm
import os
from visdom import Visdom
import time


filter_sizes = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 19]
num_filters = [100, 200, 200, 200, 200, 100, 100, 100, 100, 100, 160, 160]

batch_size=64
epochs_size=100

# 查看是否可以使用cuda
cuda=torch.cuda.is_available()
viz = Visdom()
line = viz.line(np.arange(2))
text = viz.text("<h1>Text convolution Nueral Network</h1>")

def train(corpus_,cnn,criterion,optimizer):
    best_acc=0
    time_p=[]
    # 可视化数据
    tr_acc=[]
    ev_acc=[]
    tr_mean_loss=[]
    ev_mean_loss=[]
    start_time = time.time()
    # 训练数据集
    for epoch in range(epochs_size):
        tr_loss_list=[]
        tr_acc_list=[]
        for idx in tqdm(range(0,len(corpus_.train_ids),batch_size)):
            if cuda:
                inp=Variable(torch.from_numpy(corpus_.train_ids[idx:idx+64])).cuda()
                tag=Variable(torch.from_numpy(corpus_.train_tags[idx:idx+64])).cuda()
            pred=cnn(inp)
            _,pred_idx=torch.max(pred,1)
            loss = criterion(pred,tag)
            tr_loss_list.append(loss.cpu().data.numpy())
            tr_acc_list.append((sum(pred_idx.cpu().data.numpy()==tag.cpu().data.numpy())*1./tag.size(0)))
            optimizer.zero_grad()
            loss.backward()
            # 剪裁参数梯度
            nn.utils.clip_grad_norm(cnn.parameters(), 1, norm_type=2)
            optimizer.step()
        print('epoch:{},train_mean_loss:{},train_mean_acc:{}'.format(epoch,np.mean(tr_loss_list),np.mean(tr_acc_list)))

        # 验证数据集
        ev_loss_list=[]
        ev_acc_list=[]
        for idx in tqdm(range(0,len(corpus_.eval_ids),batch_size)):
            if cuda:
                inp=Variable(torch.from_numpy(corpus_.train_ids[idx:idx+64])).cuda()
                tag=Variable(torch.from_numpy(corpus_.train_tags[idx:idx+64])).cuda()
            pred=cnn(inp)
            _,pred_idx=torch.max(pred,1)
            loss = criterion(pred,tag)
            ev_loss_list.append(loss.cpu().data.numpy())
            ev_acc_list.append((sum(pred_idx.cpu().data.numpy()==tag.cpu().data.numpy())*1./tag.size(0)))
        print('epoch:{},eval_mean_loss:{},eval_mean_acc:{}'.format(epoch,np.mean(ev_loss_list),np.mean(ev_acc_list)))

        if np.mean(ev_acc_list)>best_acc:
            torch.save(cnn,'cnn.pt')
            best_acc=np.mean(ev_acc_list)

        time_p.append(time.time()-start_time)
        tr_acc.append(np.mean(tr_acc_list))
        ev_acc.append(np.mean(ev_acc_list))
        tr_mean_loss.append(np.mean(tr_loss_list))
        ev_mean_loss.append(np.mean(ev_loss_list))
        viz.line(X=np.column_stack((np.array(time_p), np.array(time_p),np.array(time_p), np.array(time_p))),
                 Y=np.column_stack((np.array(tr_mean_loss),np.array(ev_mean_loss),np.array(tr_acc), np.array(ev_acc))),
                 win=line,
                 opts=dict(legend=["Train_mean_loss", "Eval_mean_loss", "Train_acc", "Eval_acc"]))



# 评估模型函数
def eval(corpus_,criterion,optimizer):
    # 加载模型
    cnn=torch.load('cnn.pt')
    loss_list=[]
    acc_list=[]
    for idx in tqdm(range(0,len(corpus_.eval_ids),batch_size)):
        if cuda:
            inp=Variable(torch.from_numpy(corpus_.train_ids[idx:idx+64])).cuda()
            tag=Variable(torch.from_numpy(corpus_.train_tags[idx:idx+64])).cuda()
        pred=cnn(inp)
        _,pred_idx=torch.max(pred,1)
        loss = criterion(pred,tag)
        loss_list.append(loss.cpu().data.numpy())
        acc_list.append((sum(pred_idx.cpu().data.numpy()==tag.cpu().data.numpy())*1./tag.size(0)))
    print('eval_mean_loss:{},eval_mean_acc:{}'.format(np.mean(loss_list),np.mean(acc_list)))

#预测函数
def pred(corpus_,cnn,criterion,optimizer,pred_file=''):
    pred_ids=[]
    with open(pred_file) as f:
        for line in f:
            sent_words=line.split()
            sent_ids=[corpus_.token2idx[x] for x in sent_words if x in corpus_.token2idx.keys()]
            if len(sent_ids)<2000:
                sent_ids=sent_ids+[corpus_.token2idx['<pad>']]*(2000-len(sent_ids))
            sent_ids=sent_ids[:2000]
            pred_ids.append(sent_ids)

    pred_ids=np.array(pred_ids)
    acc=[]
    for idx in tqdm(range(0,len(pred_ids),batch_size)):
        if cuda:
            inp=Variable(torch.from_numpy(pred_ids[idx:idx+64])).cuda()
            tag=Variable(torch.from_numpy(pred_ids[idx:idx+64])).cuda()
        pred=cnn(inp)
        _,pred_idx=torch.max(pred,1)
        acc.append((sum(pred_idx.cpu().data.numpy()==1)*1./tag.size(0)))
    print(np.mean(acc))



if __name__ == '__main__':
    corpus_=corpus('bank_all_0.txt','bank_all_1.txt','stop_words.csv')
    print('Max length of sents :{}'.format(np.max(corpus_.lengths)))
    print('The vocab size is {}'.format(len(corpus_.token2idx)))
    print('Train ids shape {}'.format(corpus_.train_ids.shape))
    print('Eval ids shape {}'.format(corpus_.eval_ids.shape))
    num_classes=2
    vocab_size=len(corpus_.token2idx)
    emb_dim=128
    criterion = nn.CrossEntropyLoss()

    if not os.path.isfile('cnn.pt'):
        if cuda:
            cnn=CNN(num_classes,vocab_size,emb_dim, filter_sizes, num_filters).cuda()
        else:
            cnn=CNN(num_classes,vocab_size,emb_dim, filter_sizes, num_filters).cuda()
    else:
        cnn=torch.load('cnn.pt')
    print(cnn)
    sys.exit(0)
    optimizer=optim.Adam(cnn.parameters(),lr=0.0005)
    # train(corpus_,cnn,criterion,optimizer)
    # eval(corpus_,criterion,optimizer)
    pred(corpus_,cnn,criterion,optimizer,'0428_insurance_0.txt')
