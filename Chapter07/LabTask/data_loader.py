import csv
import numpy as np
import jieba


#对于文本二分类问题的语料库类
class corpus(object):
    """docstring for ."""
    def __init__(self,):
        super(corpus,self).__init__()
        # 在字典里面先定一个pad字符
        self.token2idx={}
        self.idx2token={}
        self.sents=[]
        self.tags=[]
        self.stop_words=[]
        self.ids=[]
        self.lengths=[]
        self.words=[]
        # 加载停用词数据集
        print('Loading text data ...')
        self.load_data()
        self.sents2ids()
        self.vocab_size=len(self.idx2token)
        self.tag2idx={}
        self.idx2tag={}
        self.tag_ids=[]
        self.tag2ids()
        self.shuffle()



    # 加载文本数据,正样本和负样本
    def load_data(self):
        with open("data.csv") as f:
            lines=csv.reader(f)
            header=next(lines)
            for line in lines:
                text =list(jieba.cut(line[0]))
                if len(line) !=2 or len(text)>500 or len(text)<=10:
                    continue
                self.tags.append(line[1])
                self.sents.append(text)
                self.words.extend(text)
                self.lengths.append(len(text))
        assert len(self.sents)==len(self.tags)


    def sents2ids(self):
        self.words=list(set(self.words))
        self.words.insert(0,'<pad>')
        self.max_len=max(self.lengths)
        print("max len is {}".format(max(self.lengths)))
        print("min len is {}".format(min(self.lengths)))
        print("nums of words is :{}".format(len(self.words)))
        # 给第一个词让开位置，生成词转id和id转词词典
        self.token2idx=dict(zip(self.words,range(len(self.words))))
        self.idx2token=dict(zip(range(len(self.words)),self.words))
        for sent in self.sents:
            # 还要做一下填充，每个句子填充到2000个字，最大是2000
            sent_id=[self.token2idx[x] for x in sent if x in self.token2idx.keys()]
            sent_id=sent_id+[self.token2idx['<pad>']]*(self.max_len-len(sent_id))
            self.ids.append(sent_id)



    def tag2ids(self):
        tags=list(set(self.tags))
        self.num_class=len(tags)
        print("nums of class is :{}".format(len(tags)))
        self.tag2idx=dict(zip(tags,range(len(tags))))
        self.idx2tag=dict(zip(range(len(tags)),tags))
        for tag in self.tags:
            tag_id=self.tag2idx[tag]
            self.tag_ids.append(tag_id)

    # 打乱数据
    def shuffle(self):
        # 获得所有样本的一个索引值的list
        idxs=np.random.permutation(len(self.ids))
        print("nums of samples is {}".format(len(self.ids)))
        # 切分数据集，训练集和评估数据集,按照0.8的比率
        train_idxs=idxs[:int(len(idxs)*0.8)]
        eval_idxs=idxs[int(len(idxs)*0.8):]

        self.train_ids=np.array(self.ids)[train_idxs]
        self.train_tags=np.array(self.tag_ids)[train_idxs]

        self.eval_ids=np.array(self.ids)[eval_idxs]
        self.eval_tags=np.array(self.tag_ids)[eval_idxs]


if __name__ == '__main__':
    c=corpus()
