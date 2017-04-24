#!/usr/bin/env python
# -*- coding: utf-8 -*-
# author igor
# Created by iFantastic on 16-5-3
import os


class DocumentReader():
    def __init__(self, file_dir):
        self.file_dir = file_dir

    def __iter__(self):
        for file in os.listdir(self.file_dir):
            print("read from %s " % (os.path.join(self.file_dir, file)))
            for line in open(os.path.join(self.file_dir, file),
                             'r', encoding='utf8'):
                yield line.split()


class SingleDocumentReader():
    def __init__(self, file_path):
        self.path = file_path

    def __iter__(self):
        with open(self.path, 'r', encoding='utf8') as f:
            for line in f:
                yield line.split()


if __name__ == '__main__':
    # test Reader
    # from rnnlm.context import DATA_DIR
    # dr = DocumentReader(os.path.join(DATA_DIR,'raw'))
    # for i in dr:
    #     print(i)
    pass
