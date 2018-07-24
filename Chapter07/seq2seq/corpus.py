import numpy as np
import jieba
import sys
import codecs

#对于情感3分类的问题的语料库类
class corpus(object):
    """docstring for ."""
    def __init__(self,train_file,min_count=10):
        super(corpus,self).__init__()
        self.train_file = train_file
        self.min_count=min_count
        # 在字典里面先定一个pad字符
        self.token2idx={}
        self.idx2token={}
        self.questions=[]
        self.answers=[]
        self.stop_words=[]
        self.questions_ids=[]
        self.answers_ids=[]
        self.questions_lengths=[]
        self.answers_lengths=[]
        self.words_counts={}
        # 加载停用词数据集
        print('Loading text data ...')
        self.load_data()
        self.sents2ids()
        self.shuffle()

        print("Train data is {}".format(len(self.train_questions_ids)))
        print("Eval data is {}".format(len(self.eval_questions_ids)))
        self.vocab_size=len(self.idx2token)

    #做一个词频统计,而且词的个数太多，也得把低频词去掉
    def add_words(self,tokens):
        for t in tokens:
            self.words_counts[t]=self.words_counts.get(t,0)+1

    # 加载文本数据
    def load_data(self):
        # 对数据进行处理\
        print(self.train_file)
        with open(self.train_file) as f:
            while True:
                try:
                    # 获得下一个值:
                    question = next(f)
                    seg_question =["<s>"]+list(jieba.cut(question.strip()))+["<e>"]
                    self.questions.append(seg_question)
                    self.questions_lengths.append(len(seg_question))
                    self.add_words(seg_question)

                    answer = next(f)
                    seg_answer =["<s>"]+list(jieba.cut(answer.strip()))[:100]+["<e>"]
                    self.answers.append(seg_answer)
                    self.answers_lengths.append(len(seg_answer))
                    self.add_words(seg_answer)

                except StopIteration:
                    break

    def sents2ids(self):
        # 小于5次低频词的去掉
        words=[w for w,c in self.words_counts.items() if c>=self.min_count]
        # 再词表的最开始位置，插入一个填充符
        words.insert(0,'<pad>')
        # 给第一个词让开位置，生成词转id和id转词词典
        self.token2idx=dict(zip(words,range(len(words))))
        self.idx2token=dict(zip(range(len(words)),words))

        self.max_questions_length=max(self.questions_lengths)
        self.max_answers_length=max(self.answers_lengths)
        print("The max questions length sents is {}".format(self.max_questions_length))
        print("The max answers length sents is {}".format(self.max_answers_length))

        for sent in self.questions:
            # 还要做一下填充，每个句子填充到2000个字，最大是2000
            sent_id=[self.token2idx[x] for x in sent if x in self.token2idx.keys()]
            sent_id=sent_id+[self.token2idx['<pad>']]*(self.max_questions_length-len(sent_id))
            self.questions_ids.append(sent_id)

        for sent in self.answers:
            # 还要做一下填充，每个句子填充到2000个字，最大是2000
            sent_id=[self.token2idx[x] for x in sent if x in self.token2idx.keys()]
            sent_id=sent_id+[self.token2idx['<pad>']]*(self.max_answers_length-len(sent_id))
            self.answers_ids.append(sent_id)

    def converqus(self,questions):
        questions_ids=[]
        for q in questions:
            seg_question =["<s>"]+list(jieba.cut(q))+["<e>"]
            sent_id=[self.token2idx[x] for x in seg_question if x in self.token2idx.keys()]
            sent_id=sent_id+[self.token2idx['<pad>']]*(self.max_answers_length-len(sent_id))
            questions_ids.append(sent_id)
        return np.array(questions_ids)




    # 打乱数据
    def shuffle(self):
        assert len(self.answers_ids)==len(self.questions_ids)
        # 获得所有样本的一个索引值的list
        idxs=np.random.permutation(len(self.answers_ids))
        # 切分数据集，训练集和评估数据集,按照0.8的比率
        train_idxs=idxs[:int(len(idxs)*0.8)]
        eval_idxs=idxs[int(len(idxs)*0.8):]

        self.train_questions_ids=np.array(self.questions_ids)[train_idxs]
        self.train_answers_tags=np.array(self.answers_ids)[train_idxs]

        self.eval_questions_ids=np.array(self.questions_ids)[eval_idxs]
        self.eval_answers_ids=np.array(self.answers_ids)[eval_idxs]
if __name__ == '__main__':
    corpus=corpus("data/train/train.txt")
