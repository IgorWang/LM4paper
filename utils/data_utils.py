# -*- coding: utf-8 -*-
# Project : LM4paper
# Created by igor on 2016/11/27

import codecs
import glob
import json
import random
from collections import defaultdict

import numpy as np


class Vocabulary(object):
    def __init__(self):
        self._token_to_id = {}
        self._token_to_count = {}
        self._id_to_token = []
        self._num_tokens = 0

        self._pad_id = None
        self._s_id = None
        self._unk_id = None
        self._se_id = None  # symbol of end of sentence

    @property
    def num_tokens(self):
        return self._num_tokens

    @property
    def pad(self):
        return "<PAD>"

    @property
    def pad_id(self):
        return self._pad_id

    @property
    def se(self):
        return "</S>"

    @property
    def se_id(self):
        return self._se_id

    @property
    def unk(self):
        return "<UNK>"

    @property
    def unk_id(self):
        return self._unk_id

    @property
    def s(self):
        return "<S>"

    @property
    def s_id(self):
        return self._s_id

    def add(self, token, count):
        self._token_to_id[token] = self._num_tokens
        self._token_to_count[token] = count
        self._id_to_token.append(token)
        self._num_tokens += 1

    def finalize(self):
        self._s_id = self.get_id(self.s)
        self._unk_id = self.get_id(self.unk)
        self._pad_id = self.get_id(self.pad)
        self._se_id = self.get_id(self.se)

    def get_id(self, token):
        return self._token_to_id.get(token, self.unk_id)

    def get_token(self, id_):
        return self._id_to_token[id_]

    @staticmethod
    def from_file(filename):
        vocab = Vocabulary()
        with codecs.open(filename, "r", "utf8") as f:
            for line in f:
                word, count = line.strip().split()
                vocab.add(word, int(count))
        vocab.finalize()
        return vocab


class Dataset(object):
    def __init__(self, vocab, file_pattern, deterministic=False):
        self._vocab = vocab
        self._file_pattern = file_pattern
        self._deterministic = deterministic

    def _parse_file(self, file_name):
        print("Processing file: %s" % file_name)
        with codecs.open(file_name, "r", "utf-8") as f:
            lines = [line.strip() for line in f]
            if not self._deterministic:
                random.shuffle(lines)
            print("Finished Processing!")
            for line in lines:
                yield self._parse_sentence(line)

    def _parse_sentence(self, line):
        s_id = self._vocab.s_id
        return [s_id] + [self._vocab.get_id(word)
                         for word in line.strip().split()] + [s_id]

    def _sentence_stream(self, file_stream):
        for file_name in file_stream:
            for sentence in self._parse_file(file_name):
                yield sentence

    def _iterate(self, sentences, batch_size, num_steps):
        streams = [None] * batch_size
        x = np.zeros([batch_size, num_steps], np.int32)
        y = np.zeros([batch_size, num_steps], np.int32)
        w = np.zeros([batch_size, num_steps], np.int32)

        while True:
            x[:] = 0
            y[:] = 0
            w[:] = 0

            for i in range(batch_size):
                tokens_filled = 0

                try:
                    while tokens_filled < num_steps:
                        if streams[i] is None or len(streams[i]) <= 1:
                            streams[i] = next(sentences)
                        num_tokens = min(len(streams[i]) - 1, num_steps - tokens_filled)
                        x[i, tokens_filled:tokens_filled + num_tokens] = streams[i][:num_tokens]
                        y[i, tokens_filled:tokens_filled + num_tokens] = streams[i][1:num_tokens + 1]
                        w[i, tokens_filled:tokens_filled + num_tokens] = 1
                        streams[i] = streams[i][num_tokens:]
                        tokens_filled += num_tokens

                except StopIteration:
                    pass
            if not np.any(w):
                return
            yield x, y, w

    def iterate_once(self, batch_size, num_steps):
        def file_stream():
            for file_name in glob.glob(self._file_pattern):
                yield file_name

        for value in self._iterate(self._sentence_stream(file_stream()), batch_size, num_steps):
            yield value

    def iterate_forever(self, batch_size, num_steps):
        def file_stream():
            while True:
                file_patterns = glob.glob(self._file_pattern)
                if not self._deterministic:
                    random.shuffle(file_patterns)
                for file_name in file_patterns:
                    yield file_name

        for value in self._iterate(self._sentence_stream(file_stream()), batch_size, num_steps):
            yield value


class DatasetCharWord(object):
    def __init__(self, word_vocab, char_vocab, file_pattern, max_word_length=16, deterministic=False):
        self._word_vocab = word_vocab
        self._char_vocab = char_vocab
        self._file_pattern = file_pattern
        self._deterministic = deterministic
        self.max_word_length = max_word_length

    def _parse_file(self, file_name):
        print("Processing file: %s" % file_name)
        with codecs.open(file_name, "r", "utf-8") as f:
            lines = [line.strip() for line in f]
            if not self._deterministic:
                random.shuffle(lines)
            print("Finished Processing!")
            for line in lines:
                yield self._parse_sentence(line), self._parse_sentence_for_char(line)

    def _parse_word(self, word):
        s_id = self._char_vocab.s_id
        se_id = self._char_vocab.se_id
        pad_id = self._char_vocab.pad_id
        word = word[:self.max_word_length - 2]
        length = len(word)
        pad_length = self.max_word_length - 2 - length
        return [s_id] + [self._char_vocab.get_id(char) for char in word] + [se_id] + [pad_id] * pad_length

    def _parse_sentence_for_char(self, line):
        s_id = self._char_vocab.s_id
        se_id = self._char_vocab.se_id
        data = [self._parse_word(word) for word in line.strip().split()]
        return [[s_id] * self.max_word_length] + data + [[se_id] * self.max_word_length]

    def _parse_sentence(self, line):
        s_id = self._word_vocab.s_id
        se_id = self._word_vocab.se_id
        return [s_id] + [self._word_vocab.get_id(word)
                         for word in line.strip().split()] + [se_id]

    def _sentence_stream(self, file_stream):
        for file_name in file_stream:
            for sentence in self._parse_file(file_name):
                yield sentence

    def _iterate(self, sentences, batch_size, num_steps):
        streams_word = [None] * batch_size
        streams_char = [None] * batch_size

        word_x = np.zeros([batch_size, num_steps], np.int32)
        char_x = np.zeros([batch_size, num_steps, self.max_word_length], np.int32)
        y = np.zeros([batch_size, num_steps], np.int32)
        w = np.zeros([batch_size, num_steps], np.int32)

        while True:
            word_x[:] = 0
            y[:] = 0
            w[:] = 0

            for i in range(batch_size):
                tokens_filled = 0

                try:
                    while tokens_filled < num_steps:
                        if streams_word[i] is None or len(streams_word[i]) <= 1:
                            streams_word[i], streams_char[i] = next(sentences)
                        num_tokens = min(len(streams_word[i]) - 1, num_steps - tokens_filled)

                        word_x[i, tokens_filled:tokens_filled + num_tokens] = streams_word[i][:num_tokens]
                        char_x[i, tokens_filled:tokens_filled + num_tokens, :] = streams_char[i][:num_tokens]

                        y[i, tokens_filled:tokens_filled + num_tokens] = streams_word[i][1:num_tokens + 1]
                        w[i, tokens_filled:tokens_filled + num_tokens] = 1
                        streams_word[i] = streams_word[i][num_tokens:]
                        tokens_filled += num_tokens
                except StopIteration:
                    pass
            if not np.any(w):
                return
            yield word_x, char_x, y, w

    def iterate_once(self, batch_size, num_steps):
        def file_stream():
            for file_name in glob.glob(self._file_pattern):
                yield file_name

        for value in self._iterate(self._sentence_stream(file_stream()), batch_size, num_steps):
            yield value

    def iterate_forever(self, batch_size, num_steps):
        def file_stream():
            while True:
                file_patterns = glob.glob(self._file_pattern)
                if not self._deterministic:
                    random.shuffle(file_patterns)
                for file_name in file_patterns:
                    yield file_name

        for value in self._iterate(self._sentence_stream(file_stream()), batch_size, num_steps):
            yield value


def build_vocab(path, min_freq=3):
    frequency = defaultdict(int)
    max_length = 0
    with open(path, 'r') as f:
        for line in f:
            line = line.split()[0]
            max_length = max(max_length, len(line))
            for token in list((line)):
                frequency[token] += 1
    frequency = {k: v for k, v in frequency.items() if v >= min_freq}
    return frequency


if __name__ == '__main__':
    word_path = "/data/1b_word_vocab.txt"
    char_path = "/data/1b_char_vocab.txt"
    # counter = build_vocab(path)
    #
    # counter = sorted(counter.items(), key=lambda x: x[-1])
    # with open("/data/1b_char_vocab.txt", 'w') as f:
    #
    #     for symbols in ["<PAD>", "<UNK>", "<S>", "</S>"]:
    #         f.write(symbols)
    #         f.write(" ")
    #         f.write("0")
    #         f.write("\n")
    #
    #     for k, v in counter:
    #         f.write(k)
    #         f.write("\t")
    #         f.write(str(v))
    #         f.write("\n")

    file_pattern = "/data/one-billion-lm/training-monolingual.tokenized.shuffled/*"
    word_vocab = Vocabulary.from_file(word_path)
    char_vocab = Vocabulary.from_file(char_path)
    dataset = DatasetCharWord(word_vocab, char_vocab, file_pattern)
    iterators = dataset.iterate_once(10, 5)
    print(next(iterators))
