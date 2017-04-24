# -*- coding: utf-8 -*-
# Project : LM4paper
# Created by igor on 2016/12/12
'''
preprocessing module

Split raw text file to train and test
Build dictionary
'''
import os
import glob
import codecs
import math
from collections import defaultdict

import jieba

DEFAULT_TOKENIZER = lambda sent: list(jieba.cut(sent.strip()))
CHAR_TOKENIZER = lambda sent: list(sent.strip())


class DatasetReader():
    def __init__(self, directory, pattern, mode='word'):
        self.dir = directory
        # 原始数据必须存放在directory下的raw目录下
        self.patt = os.path.join(directory, 'raw', pattern)
        self.frequency = {}
        self.mode = mode
        if mode == "word":
            self.tokenizer = DEFAULT_TOKENIZER
        elif mode == "char":
            self.tokenizer = CHAR_TOKENIZER
        else:
            raise ValueError("wrong mode for choose tokenizer, mode must be 'word' or 'char'")

    def __iter__(self):
        files_pattern = glob.glob(self.patt)
        for filename in files_pattern:
            with codecs.open(filename, "r", "utf8") as f:
                for line in f:
                    yield line

    def build_vocab(self, min_freq=2, save=True):
        frequency = defaultdict(int)
        for sent in self:
            for token in self.tokenizer(sent):
                frequency[token] += 1
        self.frequency = {k: v for k, v in frequency.items() if v >= min_freq}
        return self.frequency

    def save_vocab(self, directory, prefix="law", add_symbols=True):
        save_path = os.path.join(directory, prefix + "_" + self.mode + ".dict")
        with codecs.open(save_path, 'w', 'utf8') as f:
            if add_symbols:  # add special symbols to dictionary
                for symbols in ["<PAD>", "<UNK>", "<S>", "</S>"]:
                    f.write(symbols)
                    f.write(" ")
                    f.write("0")
                    f.write("\n")

            for k, v in self.frequency.items():
                f.write(k)
                f.write(" ")
                f.write(str(v))
                f.write("\n")
        print("Save %s dict to %s" % (prefix, save_path))

    def build_train_test_set(self, max_lines_in_one_file=20000):
        total_lines = sum(1 for _ in self)
        print("Total lines of DataSet is %d" % total_lines)
        print("Split DataSet to Training Set and Testing Set:")

        train_lines = int(math.ceil(total_lines * 0.8))
        test_lines = total_lines - train_lines
        print("Number of training lines : %d" % train_lines)
        print("Number of test lines : %d" % test_lines)

        train_files = int(math.ceil(train_lines / max_lines_in_one_file))
        test_files = int(math.ceil(test_lines / max_lines_in_one_file))
        print("Number of training files: %d" % train_files)
        print("Number of testing files: %d" % test_files)

        # check dir
        train_files_dir = os.path.join(self.dir, 'train')
        if not os.path.exists(train_files_dir):
            os.mkdir(train_files_dir)
        else:
            for file in os.listdir(train_files_dir):
                os.remove(os.path.join(train_files_dir, file))

        test_files_dir = os.path.join(self.dir, 'test')
        if not os.path.exists(test_files_dir):
            os.mkdir(test_files_dir)
        else:
            for file in os.listdir(test_files_dir):
                os.remove(os.path.join(test_files_dir, file))

        train_files_objs = [codecs.open(
            os.path.join(self.dir, 'train', "train_%s_%04d_of_%04d.txt" % (self.mode, i, train_files)),
            'w+', 'utf8') for i in range(train_files)]
        test_files_objs = [codecs.open(
            os.path.join(self.dir, 'test', "test_%s_%04d_of_%04d.txt" % (self.mode, i, test_files)),
            'w+', 'utf8') for i in range(test_files)]

        i = 0
        j = 0
        k = 0
        for sent in self:
            if k % 5 == 0:
                test_files_objs[j].write(" ".join(self.tokenizer(sent)).strip())
                j = j + 1 if j + 1 < test_files else 0
            else:
                train_files_objs[i].write(" ".join(self.tokenizer(sent)).strip())
                i = i + 1 if i + 1 < train_files else 0
            k += 1
        for fobj in train_files_objs + test_files_objs:
            fobj.close()



if __name__ == '__main__':
    s = DEFAULT_TOKENIZER("我爱北京天安门")
    s1 = CHAR_TOKENIZER("我爱北京天安门")
    print(s1)


    dr = DatasetReader("/data/", ".*.txt", mode='char')
    counter = dr.build_vocab(min_freq=5)
    # dr.save_vocab("/data/lawdata/")
    # dr.build_train_test_set(max_lines_in_one_file=100000)

