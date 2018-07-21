# import pymongo

# db = pymongo.MongoClient().baike.items
# def texts():
#     for a in db.find(no_cursor_timeout=True).limit(1000000):
#         yield a['content']
#         
# 。。。。。。。。参考。。。。。。。。。。。。
# from sqlalchemy import *

# def sql_data_generator():
#     db = create_engine('mysql+pymysql://user:password@123.456.789.123/yourdatabase?charset=utf8')
#     result = db.execution_options(stream_results=True).execute(text('select content from articles'))
#     for t in result:
#         yield t[0]

from collections import defaultdict #defaultdict是经过封装的dict，它能够让我们设定默认值
from tqdm import tqdm #tqdm是一个非常易用的用来显示进度的库
from math import log
import re
f = open('shen_diao1.txt', 'r',encoding='UTF-8') #读取文章
s = f.read() #读取为一个字符串


class Find_Words:
    def __init__(self, min_count=10, min_pmi=0):
        self.min_count = min_count
        self.min_pmi = min_pmi
        self.chars, self.pairs = defaultdict(int), defaultdict(int) #如果键不存在，那么就用int函数
                                                                  #初始化一个值，int()的默认结果为0
        self.total = 0.
    def text_filter(self, texts): #预切断句子，以免得到太多无意义（不是中文、英文、数字）的字符串
        for a in tqdm(texts):
            for t in re.split(u'[^\u4e00-\u9fa50-9a-zA-Z]+', a): #这个正则表达式匹配的是任意非中文、
                                                              #非英文、非数字，因此它的意思就是用任
                                                              #意非中文、非英文、非数字的字符断开句子
                if t:
                    yield t
    def count(self, texts): #计数函数，计算单字出现频数、相邻两字出现的频数
        for text in self.text_filter(texts):
            self.chars[text[0]] += 1
            for i in range(len(text)-1):
                self.chars[text[i+1]] += 1
                self.pairs[text[i:i+2]] += 1
                self.total += 1
        self.chars = {i:j for i,j in self.chars.items() if j >= self.min_count} #最少频数过滤
        self.pairs = {i:j for i,j in self.pairs.items() if j >= self.min_count} #最少频数过滤
        self.strong_segments = set()
        for i,j in self.pairs.items(): #根据互信息找出比较“密切”的邻字
            _ = log(self.total*j/(self.chars[i[0]]*self.chars[i[1]]))
            if _ >= self.min_pmi:
                self.strong_segments.add(i)
    def find_words(self, texts): #根据前述结果来找词语
        self.words = defaultdict(int)
        for text in self.text_filter(texts):
            s = text[0]
            for i in range(len(text)-1):
                if text[i:i+2] in self.strong_segments: #如果比较“密切”则不断开
                    s += text[i+1]
                else:
                    self.words[s] += 1 #否则断开，前述片段作为一个词来统计
                    s = text[i+1]
        self.words = {i:j for i,j in self.words.items() if j >= self.min_count} #最后再次根据频数过滤

fw = Find_Words(16, 1)
fw.count(s)
fw.find_words(s)

import pandas as pd
words = pd.Series(fw.words).sort_values(ascending=False)
print(words)
# pd.DataFrame(words).to_csv('result_shendiao1.txt', header = False)