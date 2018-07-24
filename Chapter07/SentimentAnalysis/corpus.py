import numpy as np
import jieba
import sys

#对于情感3分类的问题的语料库类
class corpus(object):
    """docstring for ."""
    def __init__(self, pos_file,neg_file,neu_file,stop_words_file,min_count=15):
        super(corpus,self).__init__()
        self.pos_file = pos_file
        self.neg_file = neg_file
        self.neu_file = neu_file
        self.min_count=min_count
        # 在字典里面先定一个pad字符
        self.token2idx={}
        self.idx2token={}
        self.sents=[]
        self.tags=[]
        self.stop_words=[]
        self.ids=[]
        self.lengths=[]
        self.words_counts={}
        # 加载停用词数据集
        self.load_stop_words(stop_words_file)
        print('Loading text data ...')
        self.load_data()
        self.sents2ids()
        self.shuffle()

        print("Train data is {}".format(len(self.train_ids)))
        print("Eval data is {}".format(len(self.eval_ids)))
        self.num_classes=3
        self.vocab_size=len(self.idx2token)

    #做一个词频统计,而且词的个数太多，也得把低频词去掉
    def add_words(self,tokens):
        for t in tokens:
            self.words_counts[t]=self.words_counts.get(t,0)+1

    # 加载文本数据
    def load_data(self):

        # 对正样本的处理,标记为0标签
        with open(self.pos_file) as f:
            for line in f:
                seg_list =list(jieba.cut(line))
                # 对每一行记录去掉停用词
                tokens=[x for x in line if x not in self.stop_words]
                self.sents.append(tokens)
                self.tags.append(0)
                self.lengths.append(len(tokens))
                self.add_words(tokens)


        # 对负样本的饿处理，标记为1标签
        with open(self.neg_file) as f:
            for line in f:
                tokens=[x for x in line if x not in self.stop_words]
                self.sents.append(tokens)
                self.tags.append(1)
                self.lengths.append(len(tokens))
                self.add_words(tokens)

       # 对中立样本的饿处理，标记为2标签
        with open(self.neu_file) as f:
            for line in f:
                # 对每一行记录去掉停用词
                tokens=[x for x in line.strip().split() if x not in self.stop_words]
                self.sents.append(tokens)
                self.tags.append(2)
                self.lengths.append(len(tokens))
                self.add_words(tokens)


    def sents2ids(self):
        # 小于5次低频词的去掉
        words=[w for w,c in self.words_counts.items() if c>=self.min_count]
        # 再词表的最开始位置，插入一个填充符
        words.insert(0,'<pad>')
        # 给第一个词让开位置，生成词转id和id转词词典
        self.token2idx=dict(zip(words,range(len(words))))
        self.idx2token=dict(zip(range(len(words)),words))

        self.max_length=max(self.lengths)
        print("The max length sents is {}".format(self.max_length))
        for sent in self.sents:
            # 还要做一下填充，每个句子填充到2000个字，最大是2000
            sent_id=[self.token2idx[x] for x in sent if x in self.token2idx.keys()]
            sent_id=sent_id+[self.token2idx['<pad>']]*(self.max_length-len(sent_id))
            self.ids.append(sent_id)

    def load_stop_words(self,stop_words_file):
        with open(stop_words_file) as f:
            for word in f:
                self.stop_words.append(word.strip())
    # 打乱数据
    def shuffle(self):
        # 获得所有样本的一个索引值的list
        idxs=np.random.permutation(len(self.ids))
        # 切分数据集，训练集和评估数据集,按照0.8的比率
        train_idxs=idxs[:int(len(idxs)*0.8)]
        eval_idxs=idxs[int(len(idxs)*0.8):]

        self.train_ids=np.array(self.ids)[train_idxs]
        self.train_tags=np.array(self.tags)[train_idxs]

        self.eval_ids=np.array(self.ids)[eval_idxs]
        self.eval_tags=np.array(self.tags)[eval_idxs]
if __name__ == '__main__':
    corpus=corpus("data/pos.csv","data/neg.csv","data/neutral.csv","data/stop_words.csv")
