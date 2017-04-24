#!/usr/bin/env python
# -*- coding: utf-8 -*-
# author igor
# Created by iFantastic on 16-5-3
'''
预处理模块：
 - 读取原始数据
 - 构建词典
 - 载入词典
 - 更新词典
 - 生成训练,评价数据 id模式
 - mini-batch
'''
import os
import pickle
import random
import linecache
import csv
from collections import defaultdict

from gensim import corpora
import numpy as np

from lmpaper.context import DATA_DIR
from lmpaper.utils import DocumentReader

MODEL_PATH = os.path.join(DATA_DIR, 'model')
EXAMPLE_NUMS = 680000
unk_token = 'UNK'


def load_dictionary(model_file_path):
    '''
    加载字典
    :param model_file_path:
    :return:
    '''
    if os.path.exists(model_file_path):
        with open(model_file_path, 'rb') as f:
            dictionary = pickle.load(f)
        f.close()
        print("read dictionary from %s" % (model_file_path))
        default_dict = defaultdict(lambda: dictionary['UNK'])
        default_dict.update(dictionary)
        default_dict.update({'PAD': max(dictionary.values()) + 1})
        return default_dict
    else:
        raise FileExistsError


def build_dictionary(data_dir, dict_name, min_freq=5, load=True):
    '''
    构建词典
    :param data_dir:数据路径
    :param min_freq:最小频次
    :return:
    '''
    dictionary_path = os.path.join(MODEL_PATH, dict_name)
    if load:
        if os.path.exists(dictionary_path):
            with open(dictionary_path, 'rb') as f:
                dictionary = pickle.load(f)
            f.close()
            print("read dictionary from %s" % (dictionary_path))
            return dictionary
    dr = DocumentReader(data_dir)
    dictionary = corpora.Dictionary(doc for doc in dr)
    # 去除低频的ID
    filter_ids = [tokenid for tokenid, docfreq in dictionary.dfs.items() if docfreq < min_freq]
    dictionary.filter_tokens(filter_ids)
    dictionary.compactify()

    pickle.dump(dictionary.token2id, open(dictionary_path, 'wb'))
    print("SVAE dictionary to %s" % (dictionary_path))

    return dictionary.token2id


def doc_to_id(doc, token2id):
    '''
    讲文本映射为ID
    :param doc:list of words
    :return:list of ids
    '''
    dictionary = defaultdict(lambda: token2id['UNK'])
    dictionary.update(token2id)
    return [dictionary[i] for i in doc]


def iterator(token2id, raw_data, batch_size, num_steps):
    '''
    :param raw_data:原始数据 List of words
    :param batch_size: 批量大小
    :param num_steps: 展开步数
    :yields: [batch_size,num_steps]
    '''

    raw_data = doc_to_id(raw_data, token2id)  # words to id

    data_len = len(raw_data)
    batch_len = data_len // batch_size
    data = np.zeros([batch_size, batch_len], dtype=np.int32)
    for i in range(batch_size):
        data[i] = raw_data[batch_len * i:batch_len * (i + 1)]

    epoch_size = (batch_len - 1) // num_steps

    for i in range(epoch_size):
        x = data[:, i * num_steps:(i + 1) * num_steps]
        y = data[:, i * num_steps + 1:(i + 1) * num_steps + 1]
        yield (x, y)


def random_read(batch_size, raw_path, max_epoch, multiple=1):
    '''
    随机读取数据中的n个样本
    :param batch_size :批量大小
    :param raw_path : 数据文件的路径
    :param max_epoch : 最大的迭代次数
    :param multiple: 抽取批量多少倍的数据
    :return:
    '''
    print("cache data...maybe cost some minutes...")
    linecache.getline(raw_path, 0)
    for j in range(max_epoch):
        raw_data = []
        for i in range(batch_size * multiple):
            offset = random.randrange(EXAMPLE_NUMS)
            example = linecache.getline(raw_path, offset)
            raw_data.extend(example.split())
        yield raw_data


# def file_to_id(raw_dir, train_dir, filename, token2id):
#     dr = DocumentReader(raw_dir)
#     with open(os.path.join(train_dir, filename), 'wb') as f:
#         i = 0
#         for doc in dr:
#             doc_id = doc_to_id(doc, token2id)
#
#             i += 1
#             if i % 10000 == 0:
#                 print("proprecess %d docment" % (i))
#     f.close()

def main():
    # 构造词典

    # 指明文件所在的目录
    path = os.path.join(DATA_DIR, 'text')
    # 字典的名称
    file_name = "dict.pickle"

    # 构造字典
    token2id = build_dictionary(path, dict_name=file_name, load=False)
    print(len(token2id))
    token2id.update(({"PAD": max(token2id.values()) + 1}))
    print("The total of tokens is %s" % len(token2id))


def file_to_id(save_file, dr):
    token2id = load_dictionary(os.path.join(DATA_DIR, 'model', 'dict.pickle'))
    with open(save_file, 'w', encoding='utf8') as csvfile:
        csv_writer = csv.writer(csvfile, delimiter=",")
        for sent in dr:
            result = [token2id[token] for token in sent]
            pads = [token2id['PAD']] * (50 - len(result))
            result.extend(pads)
            csv_writer.writerow(result)


def file_to_id_with_buckets(save_dir, dr, buckets=None):
    # defalut_buckets = [15, 25, 35, 50]
    token2id = load_dictionary(os.path.join(DATA_DIR, 'model', 'dict.pickle'))
    if buckets is None:
        buckets = [15, 25, 35, 50]
    file_path = [os.path.join(save_dir, 'train_%s.csv' % (str(i))) for i in buckets]
    file_list = [open(i, 'w') for i in file_path]
    csv_writers = [csv.writer(i, delimiter=",") for i in file_list]
    intervals = [(0, buckets[0])] + list(zip(buckets[:-1], buckets[1:]))
    for line in dr:
        length = len(line)
        for iv in intervals:
            if length in range(iv[0], iv[1]):
                index = intervals.index(iv)
                break
        pad_length = buckets[index] - length
        pads = [token2id['PAD']] * pad_length
        line = [token2id[token] for token in line]
        line.extend(pads)
        csv_writers[index].writerow(line)
    for file in file_list:
        file.close()


if __name__ == '__main__':
    # 文件到id 将文件存储为id,并且添加PAD
    from lmpaper.utils import SingleDocumentReader

    dr = SingleDocumentReader(os.path.join(DATA_DIR, 'text', 'tokens.txt'))
    # file_to_id(os.path.join(DATA_DIR, 'test', 'test.csv'), dr)
    file_to_id_with_buckets(os.path.join(DATA_DIR, 'train'), dr)

    # main()
    # raw_dir = os.path.join(DATA_DIR, 'text')
    # token2id = build_dictionary(os.path.join(raw_dir), dict_name='dict.pickle', load=False)
    # token2id = build_dictionary(data_dir=raw_dir, min_freq=10, load=False)
    # for i in random_read(1, os.path.join(raw_dir, 'text.txt'), 10):
    #     print(i)
    # print(len(token2id))
    # file_to_id(raw_dir, train_dir, 'text.txt', token2id)
    # Test doc_to_id pass
    # token2id = {"I": 0, "Go": 1, "UNK": 2}
    # print(doc_to_id(['I',"Get"],token2id))
    # iterator(token2id, raw_dir, 10, 5)
    token2id = load_dictionary(os.path.join(DATA_DIR, 'model', 'dict.pickle'))
    print(len(token2id))
    # print(token2id['UNK'])
    # print(token2id['PAD'])
    # dr = DocumentReader(raw_dir)
    # for i in dr:
    #     print(i)
