#!/usr/bin/env
# coding:utf-8
"""
Created on 2018/7/15 10:39

base Info
"""
__author__ = 'xx'
__version__ = '1.0'



import torch
#
# str = "one Created on 2018/7/15 10:39"
#
# print(str.split('01'))   # 根据字符切开
# print(str.strip('o23454679'))   #去掉两端 出现过的字符
#
# s = '---123+++'
#
# print( s.strip('-+') )
#
# print( s.count('-') )
#
# print('{0}'.format("test"))
#
#
# print( s.maketrans(1) )

# demo = str()
# print(type(demo))
# print("demo = ",demo)
# # demo.__len__()
# demo += "xx"
# print(id(demo))
# demo = 'xasd'
# # print(demo.__len__())
# # print(id(demo))
# #
# # print( demo.__add__("xx") )
# print( demo.__contains__('xsa'))
#
# print(help(demo.__getattribute__))
# print(demo.__getattribute__())
# print(demo)


# demo_str = "若有人兮山之阿"
# print( demo_str.__len__() )
#
# print( demo_str.__dir__() )
# print( demo_str.__eq__("若有人兮山之阿"))
# print( demo_str.__mul__(3))
# print( demo_str.__sizeof__())
# print( demo_str.__hash__())
# print(id(demo_str))
# print(demo_str.__repr__())
# print(demo_str.__str__())
# # print('若有人兮山之'.__sizeof__())
# # iter_demo =  demo_str.__iter__()
# #
# # print(type(iter_demo))
#
# print(id("1"))
# print(id("2"))
#
# print(type(id("1")))

# import sys
# print(sys.version)
#
# test1 = "python字符串的不可变性"
# test2 = "python字符串的不可变性"
# print("test1 id = {0}, test2 id = {1} ".format(str(id(test1)), str(id(test2))))
# print(test1 == test2)
# print(id(test1) == id(test2))


# 一种字符串修改方式
# str = "immutable"
# modified_string = str.replace('im','')
# print(modified_string)

# 一种字符串修改方式
# str = "imm"
#
# modified_string_1 = str.replace('im','')
# print(modified_string_1)
# modified_string_2 = str[2:]
# print(modified_string_2)
# print(id(str), id(modified_string_1), id(modified_string_2))
#
#
# for i in range(2):
#     print(i)




#按空格分词
# str = 'Israel has carried out its biggest attack against Hamas militant targets in'\
#         'Gaza since the war in 2014, Prime Minister Benjamin Netanyahu says'
#
# list_str = str.split(' ')
# # print(list_str)
#
# # 按条件查找
# #  计算每个单词中 i 出现的次数
# counts = []
# for word in list_str:
#     counts.append( (word, word.count('i')) )
#
# res = map(lambda word:  word.count('i'), list_str)  # python2 返回列表   python3 返回迭代器
# print(res)
#
# print(str(12))
# print(str(1))
# demo_list = [1,2,3,4,5]
# print(str(demo_list))
# import ast
# s = '{"host":"192.168.11.22", "port":3306, "user":"abc",\
#               "passwd":"123", "db":"mydb", "connect_timeout":10}'
#
# print(ast.literal_eval(s))

# import json
# import ast
# s = '{"host":"192.168.11.22", "port":3306, "user":"abc",\
#               "passwd":"123", "db":"mydb", "connect_timeout":10}'
#
# #string -> dict
# d = ast.literal_eval(s)
#
# print(type( json.dumps(d)))

# import lxml
# from lxml import etree
# xml_string = '<root>' \
#                 '<foo id="foo-id" class="foo zoo">Foo</foo>' \
#                 '<bar>中文</bar>' \
#                 '<baz></baz>' \
#              '</root>'
#
# root = etree.fromstring(xml_string)
# doms = root.getchildren()
# foo = doms[0]
# print(foo.text, foo.attrib)
#

#
# class Person:
#     def __init__(self):
#         self.name = "xxx"
#         self.age = "666"
#
#     def __str__(self):
#         return "姓名为：{} 年龄为: {}".format(self.name, self.age)
#
# xx = Person()
# print( str(xx) )
# print( xx.__str__())


# import re

# re.split()
# re.search()
# re.match()
# re.findall()
# re.sub()   #替换
# re.compile()

# import re
# s = "dog runs to cat. I run to dog."
# pattern = r"run"       #匹配模式
# res = re.search(pattern, s)
# print(res.span())

#
# s = '<w>这次</w><w>代表大会</w><w>是</w><w>在</w><w><NAMEX TYPE="LOCATION">中国</NAMEX></w><w>改革开放</w><w>和</w><w>社会主义</w><w>现代化</w><w>建设</w><w>发展</w><w>的</w><w>关键</w><w>时刻</w><w>召开</w><w>的</w><w>历史性</w><w>会议</w><w>。</w>'
# # pattern = r'</?w>'
# # res = re.sub(pattern,"", s)
# # print(res)
#
# s = '<w>这次</w><w>代表大会</w><w>是</w><w>在</w><w><NAMEX TYPE="LOCATION">中国</NAMEX></w><w>改革开放</w>'
# pattern = r'\b((?!<w>)\w)+\b'
# # pattern = r"(?!</?w>)+<.*>(?!</?w>)+"
# res = re.search(pattern, s)
# print(res.span())
# print(res.group())
#
#
#
#
# s = "aswerabc324 234"
# p = r"\b((?!abc)\w)+\b"
# res = re.search(p, s)
# print(res)
#



# s = "France win World Cup 2018 final in breathless six-goal thriller against Croatia"
# pattern = r"\w*i\w*"
# res = re.search(pattern, s)
# print(res.group())
#
# import torch
# import numpy as np
# from torch.autograd import Variable
# # tensor = torch.Tensor([1,2,3,4])
# # print(tensor - 2)
# # np_data = np.array([2,23,4,5])
# # print(tensor - np_data)
# # print( np_data)
#
#
# tensor = torch.FloatTensor([[1,2],[3,4]])
# variable = Variable(tensor, requires_grad=True)
#
# print(tensor)
# print(variable.data)










# import torch
# import torch.nn.functional as F
# from torch.autograd import Variable
# import matplotlib.pyplot as plt
#
# # fake data
# x = torch.linspace(-5, 5, 200)  # x data (tensor), shape=(100, 1)
# # print(x)
# x = Variable(x)
# x_np = x.data.numpy()   # numpy array for plotting
#
# # following are popular activation functions
# y_relu = F.relu(x).data.numpy()
# # print(y_relu)
# y_sigmoid = F.sigmoid(x).data.numpy()
# y_tanh = F.tanh(x).data.numpy()
# # y_softplus = F.softplus(x).data.numpy()
#
# prelu = torch.nn.PReLU()
# y_prelu = prelu(x).data.numpy()
# # y_softmax = F.softmax(x)  softmax is a special kind of activation function, it is about probability
# # F.prelu(x).data.numpy()
#
#
# # plt to visualize these activation function
# plt.figure(1, figsize=(8, 6))
# plt.subplot(221)
# plt.plot(x_np, y_relu, c='red', label='relu')
# plt.ylim((-1, 5))
# plt.legend(loc='best')
#
# plt.subplot(222)
# plt.plot(x_np, y_sigmoid, c='red', label='sigmoid')
# plt.ylim((-0.2, 1.2))
# plt.legend(loc='best')
#
# plt.subplot(223)
# plt.plot(x_np, y_tanh, c='red', label='tanh')
# plt.ylim((-1.2, 1.2))
# plt.legend(loc='best')
#
# plt.subplot(224)
# plt.plot(x_np, y_prelu, c='red', label='y_prelu')
# plt.ylim((-2, 6))
# plt.legend(loc='best')
#
# plt.show()


# from snownlp import SnowNLP
# text = u'''
# 自然语言处理是计算机科学领域与人工智能领域中的一个重要方向。
# 它研究能实现人与计算机之间用自然语言进行有效通信的各种理论和方法。
# 自然语言处理是一门融语言学、计算机科学、数学于一体的科学。
# 因此，这一领域的研究将涉及自然语言，即人们日常使用的语言，
# 所以它与语言学的研究有着密切的联系，但又有重要的区别。
# 自然语言处理并不是一般地研究自然语言，
# 而在于研制能有效地实现自然语言通信的计算机系统，
# 特别是其中的软件系统。因而它是计算机科学的一部分。
# '''
#
# s = SnowNLP(text)
#
# print(s.keywords(3))
# print(s.summary(3))
# print(s.sentences)
# print(s.pinyin)
# zip_tag = s.tags
# for i in zip_tag:
#     print(i)
# print(s.tags)
# print(s.tf)
# print(s.idf)
# print(s.sim([u'自然']))
# print(s.words)

# print(s.han)

#
# import thulac
#
# thu1 = thulac.thulac()  #默认模式
# text = thu1.cut("我爱北京天安门", text=True)  #进行一句话分词
# print(text)
# import _pickle as cPickle
# print(cPickle)


# from xmnlp import xmnlp

# doc = """自然语言处理: 是人工智能和语言学领域的分支学科。
# 在这此领域中探讨如何处理及运用自然语言；自然语言认知则是指让电脑“懂”人类的语言。
# 自然语言生成系统把计算机数据转化为自然语言。自然语言理解系统把自然语言转化为计算机程序更易于处理的形式。"""
#
# xm = xmnlp(doc)
# xm.set_userdict('/path/to/userdict.txt')
# print(xm.seg(hmm=True))
#
# '''
# xm  = XmNLP()
# xm.set_userdict('/path/to/userdict.txt')
# print(xm.seg(doc, hmm=True))
# '''

# from xmnlp import xmnlp as xm
# from xmnlp xmnlp as xm
#
# doc = """自然语言处理: 是人工智能和语言学领域的分支学科。
# 在这此领域中探讨如何处理及运用自然语言；自然语言认知则是指让电脑“懂”人类的语言。
# 自然语言生成系统把计算机数据转化为自然语言。自然语言理解系统把自然语言转化为计算机程序更易于处理的形式。"""
# xm.set_userdict('/path/to/userdict.txt')
# print(xm.seg(doc, hmm=True))


import re


def rep(matched):
    #     print(matched.group('tag'))
    tag = matched.group('tag')
    if (len(tag) == 3):
        return '-'
    else:
        return ""


s = '<w>到</w><w>机场</w><w>迎接</w><w><NAMEX TYPE="PERSON">江</NAMEX></w><w>主席</w><w>的</w><w><NAMEX TYPE="LOCATION">美</NAMEX></w><w>方</w><w>人员</w><w>有</w><w><NAMEX TYPE="LOCATION">马萨诸塞州</NAMEX></w><w>州长</w><w>和</w><w><NAMEX TYPE="LOCATION">波士顿</NAMEX></w><w>市长</w><w>等</w><w>。</w>'

# pattern = '(?P<tag></?w>)'
pattern = '<.*?>'

# res = re.sub(pattern,rep, s)        #rep 可以是函数  也可以是字符串
res = re.sub(pattern, "", s)
print(res)