#!/usr/bin/env python
# -*- coding: utf-8 -*-
# author igor
# Created by iFantastic on 16-5-6
'''
模型预测
'''

from collections import defaultdict

import tensorflow as tf
import numpy as np
from scipy.spatial import distance
from tensorflow.python.ops import array_ops

from lmpaper.context import *
from lmpaper.model import inference
from lmpaper.proprecess import load_dictionary, random_read, iterator
from lmpaper.reader import distorted_inputs

tf.app.flags.DEFINE_string('checkpoint_dir', os.path.join(DATA_DIR, 'model'), 'checkpoint_directroy')


def evaluate(input_x, config):
    '''
    预测
    :param input_x:
    :param input_y:
    :return:predict_y,final_state
    '''
    graph = tf.Graph()
    with graph.as_default(), tf.Session() as sess:
        features = tf.placeholder(dtype=tf.int32, shape=[config.batch_size,
                                                         config.num_steps], name='input_words')
        # targets = tf.placeholder(dtype=tf.int32, shape=[config.batch_size,
        #                                                 config.num_steps], name='target_words')

        logits, init_state, final_state = inference(features, False, config)

        argmax = tf.arg_max(logits, dimension=1)

        saver = tf.train.Saver()

        ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
            print("restore model from %s" % ckpt.model_checkpoint_path)
        else:
            print("No checkpoint file found")

        state = init_state.eval()
        feed_dict = {features: input_x, init_state: state}

        predict, embedding = sess.run([argmax, final_state], feed_dict)
        embedding, _ = array_ops.split(1, 2, state)
    return predict, embedding


class LanguageModel(object):
    def __init__(self, config):
        # build graph
        self.sess = tf.Session()
        self.graph = tf.Graph()
        self.config = config
        self.features = tf.placeholder(dtype=tf.int32, shape=[config.batch_size,
                                                              config.num_steps], name='input_words')

        self.logits, self.init_state, self.final_state = inference(self.features, False, config)
        self.argmax = tf.arg_max(self.logits, dimension=1)
        self.embedding, _ = array_ops.split(1, 2, self.final_state)
        saver = tf.train.Saver()
        ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(self.sess, ckpt.model_checkpoint_path)
            print("restore model from %s" % ckpt.model_checkpoint_path)
        else:
            print("No checkpoint file found")

            # dictionary_path = os.path.join(FLAGS.model_dir, 'dict.pickle')
            # token2id = load_dictionary(dictionary_path)
            # self.token2id = defaultdict(lambda: self.token2id["UNK"])
            # self.token2id.update(token2id)
            # self.id2token = {v: k for k, v in self.token2id.items()}

    def predict(self, input_x):
        state = self.init_state.eval(session=self.sess)
        feed_dict = {self.features: input_x, self.init_state: state}
        predict, embedding = self.sess.run([self.argmax, self.embedding], feed_dict=feed_dict)
        return predict, embedding

        # def sample_a_sentence(self, prime=None):
        #     unk = self.token2id['UNK']
        #     input_x = [unk] * config.num_steps
        #     if prime is None:
        #         input_x[0] = self.token2id['原告']
        #         prime = ["原告"]
        #     else:
        #         for i in range(len(prime)):
        #             input_x[i] = self.token2id[prime[i]]
        #     input_x = np.asarray(input_x, dtype=np.int32)
        #     for step in range(len(prime), config.num_steps):
        #         predict, _ = self.predict([input_x])
        #         input_x[step] = predict[step]
        #     return input_x


if __name__ == '__main__':
    dictionary_path = os.path.join(FLAGS.model_dir, 'dict.pickle')
    token2id = load_dictionary(dictionary_path)
    id2token = {v: k for k, v in token2id.items()}
    config = TestConfig()
    config.vocab_size = len(token2id)
    filenames = [os.path.join(DATA_DIR, 'train', 'train.csv')]

    x, y = distorted_inputs(filenames, config.batch_size)
    lm = LanguageModel(config)
    threads = tf.train.start_queue_runners(sess=lm.sess)
    sent = []
    embeddings = []
    for i in range(10):
        input_x, _ = lm.sess.run([x, y])
        sent.append(''.join([id2token[i] for i in input_x[0]]))
        result, embedding = lm.predict(input_x)
        print("x:", [id2token[i] for i in input_x[0]])
        print("predict:", [id2token[i] for i in result])
        embeddings.append(embedding[0])
        print('\n')

    for i in range(10):
        for j in range(i + 1, 10):
            print(sent[i])
            print(sent[j])
            print(1 - distance.cosine(embeddings[i], embeddings[j]))

            #
            # # print([id2token[i] for i in lm.sample_a_sentence(prime=['没有','履行'])])
            # for data in train_data:
            #     for x, y in iterator(token2id, data, config.batch_size, config.num_steps):
            #         predict, _ = lm.predict(x)
            #         print([id2token[i] for i in y[0]])
            #         print([id2token[i] for i in predict])
            #         print('--------')
            #         # print(y)
