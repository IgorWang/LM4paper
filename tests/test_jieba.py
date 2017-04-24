# -*- coding: utf-8 -*-
# Project : LM4paper
# Created by igor on 16-12-23

import time

import jieba

jieba.enable_parallel(10)

data_path= '/data/lawdata/raw/raw.document.txt'
t1 = time.time()
content = open(data_path,"rb").read()
words = " ".join(jieba.cut(content))
t2 = time.time()
tm_cost = t2 - t1
log_f = open("/data/lawdata/raw/raw.tokenized.document.txt",'wb')
log_f.write(words.encode('utf8'))

print('speed %s bytes/second' % (len(content)/tm_cost))