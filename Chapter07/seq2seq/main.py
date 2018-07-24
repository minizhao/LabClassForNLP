import torch
from corpus import corpus
import numpy as np
from model import Seq2Seq


import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import sys
from tqdm import tqdm
import os
from visdom import Visdom
import time
import os
import pickle as pkl



epochs_size=20
batch_size=32
cuda=torch.cuda.is_available()


def train(corpus,model):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0002)
    criterion = nn.CrossEntropyLoss()
    for epoch in range(epochs_size):
        train_loss=[]
        count=0
        for idx in tqdm(range(0,len(corpus.train_questions_ids),batch_size)):
            if cuda:
                questions=Variable(torch.from_numpy(corpus.train_questions_ids[idx:idx+batch_size])).cuda()
                answers=Variable(torch.from_numpy(corpus.train_answers_tags[idx:idx+batch_size])).cuda()
            if questions.size(0)!=batch_size:
                continue
            pred=model(questions,answers,corpus)
            answers=answers.view(-1)
            loss = criterion(pred,answers)
            optimizer.zero_grad()
            loss.backward()
            # 剪裁参数梯度
            nn.utils.clip_grad_norm(model.parameters(), 1, norm_type=2)
            optimizer.step()
            train_loss.append(loss.data[0])
            count+=1
            if count%50==0:
                torch.save(model,"saves/model.pkl")
                print("epoch:{},step:{},mean_loss:{}".format(epoch,count,np.mean(train_loss)))

def test(corpus,model):
    questions=["缘分是可以相信的么?","罐头豆豉鱼怎样弄好吃?"]
    questions_ids=corpus.converqus(questions)
    answers=None
    if cuda:
        questions=Variable(torch.from_numpy(questions_ids)).cuda()
    pred=model(questions,answers,corpus,train=False)
    print(pred)


if __name__ == '__main__':

    if os.path.isfile("saves/train.pkl"):
        corpus=pkl.load(open("saves/train.pkl",'rb'))
    else:
        corpus=corpus("data/train/train.txt.1")
        pkl.dump(corpus,open('saves/train.pkl','wb'))
        print("save the corpus file")


    if os.path.isfile("saves/model.pkl"):
        model=torch.load("saves/model.pkl")
    else:
        model=Seq2Seq(corpus.vocab_size)

    if cuda:
        model=model.cuda()

    test(corpus,model)
