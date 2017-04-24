# -*- coding: utf-8 -*-
# Project : LM4paper
# Created by igor on 17-3-14
import os
import sys
import time
import json

import numpy as np
import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector

from  mixlm.lm_train import *
from mixlm.clstmdnn import CLSTMDNN
from bmlm.common import CheckpointLoader


def load_from_checkpoint(saver, logdir):
    sess = tf.get_default_session()
    ckpt = tf.train.get_checkpoint_state(logdir)
    if ckpt and ckpt.model_checkpoint_path:
        if os.path.isabs(ckpt.model_checkpoint_path):
            saver.restore(sess, ckpt.model_checkpoint_path)
        else:
            saver.restore(sess, os.path.join(logdir, ckpt.model_checkpoint_path))
        return True
    return False


class Model():
    def __init__(self, logdir):
        hps = CLSTMDNN.get_default_hparams().parse(FLAGS.hpconfig)
        hps.num_gpus = FLAGS.num_gpus
        hps.batch_size = 1

        self.word_vocab = Vocabulary.from_file(os.path.join(FLAGS.vocabdir, "1b_word_vocab.txt"))
        self.char_vocab = Vocabulary.from_file(os.path.join(FLAGS.vocabdir, "1b_char_vocab.txt"))

        with tf.variable_scope("model"):
            hps.num_sampled = 0
            hps.keep_prob = 1.0
            self.model = CLSTMDNN(hps, "eval", "/cpu:0")

        if hps.average_params:
            print("Averaging parameters for evaluation.")
            self.saver = tf.train.Saver(self.model.avg_dict)
        else:
            self.saver = tf.train.Saver()

        # Use only 4 threads for the evaluation
        config = tf.ConfigProto(allow_soft_placement=True,
                                intra_op_parallelism_threads=20,
                                inter_op_parallelism_threads=1)

        self.sess = tf.Session(config=config)

        with self.sess.as_default():
            if load_from_checkpoint(self.saver, logdir):
                global_step = self.model.global_step.eval()
                print("Successfully loaded model at step=%s." % global_step)
            else:
                print("Can't restore model from %s" % logdir)

        self.hps = hps

    def get_char_embedding(self, char):
        id = self.char_vocab.get_id(char)
        x = np.zeros(shape=(4, 20, 16))
        x[:, :, :] = id
        vector = self.sess.run([self.model.char_embedding.outputs],
                               feed_dict={self.model.char_x: x})
        # print(self.model.char_embedding)

        return vector[0][0][0]

    def get_word_embedding(self, word):
        id = self.word_vocab.get_id(word)
        x = np.zeros(shape=(4, 20))
        x[:, :] = id
        vector = self.sess.run([self.model.word_embedding.outputs],
                               feed_dict={self.model.word_x: x})

        return vector[0][0][0]


def visualize_char(model, path="/home/aegis/igor/LM4paper/tests/textchar.txt", ):
    chars = open(path, 'r').read().splitlines()
    embedding = np.empty(shape=(len(chars), model.hps.emb_char_size), dtype=np.float32)
    for i, char in enumerate(chars):
        embedding[i] = model.get_char_embedding(char)
    print(embedding)
    print(embedding.shape)

    logdir = "/data/visualog/char/"
    metadata = os.path.join(logdir, "metadata.tsv")

    with open(metadata, "w") as metadata_file:
        for c in chars:
            metadata_file.write("%s\n" % c)

    tf.reset_default_graph()
    with tf.Session() as sess:
        X = tf.Variable([0.0], name='embedding')
        place = tf.placeholder(tf.float32, shape=embedding.shape)
        set_x = tf.assign(X, place, validate_shape=False)
        sess.run(tf.global_variables_initializer())
        sess.run(set_x, feed_dict={place: embedding})

        saver = tf.train.Saver([X])

        saver.save(sess, os.path.join(logdir, 'char.ckpt'))

        config = projector.ProjectorConfig()
        # One can add multiple embeddings.
        embedding = config.embeddings.add()
        embedding.tensor_name = X.name
        # Link this tensor to its metadata file (e.g. labels).
        embedding.metadata_path = metadata
        # Saves a config file that TensorBoard will read during startup.
        projector.visualize_embeddings(tf.summary.FileWriter(logdir), config)


def visualize_word(model, path="/home/aegis/igor/LM4paper/tests/testdata.txt"):
    words = open(path, 'r').read().splitlines()
    embedding = np.empty(shape=(len(words), model.hps.emb_word_size), dtype=np.float32)
    for i, w in enumerate(words):
        embedding[i] = model.get_word_embedding(w)
    print(embedding)
    print(embedding.shape)

    logdir = "/data/visualog/word/"
    metadata = os.path.join(logdir, "metadata.tsv")

    with open(metadata, "w") as metadata_file:
        for w in words:
            metadata_file.write("%s\n" % w)

    tf.reset_default_graph()
    with tf.Session() as sess:
        X = tf.Variable([0.0], name='embedding')
        place = tf.placeholder(tf.float32, shape=embedding.shape)
        set_x = tf.assign(X, place, validate_shape=False)
        sess.run(tf.global_variables_initializer())
        sess.run(set_x, feed_dict={place: embedding})

        saver = tf.train.Saver([X])

        saver.save(sess, os.path.join(logdir, 'word.ckpt'))

        config = projector.ProjectorConfig()
        # One can add multiple embeddings.
        embedding = config.embeddings.add()
        embedding.tensor_name = X.name
        # Link this tensor to its metadata file (e.g. labels).
        embedding.metadata_path = metadata
        # Saves a config file that TensorBoard will read during startup.
        projector.visualize_embeddings(tf.summary.FileWriter(logdir), config)


if __name__ == '__main__':
    model = Model(logdir="/data/lmlog/train")
    # vector = model.get_word_embedding("hello")
    # print(vector)
    visualize_word(model, path="/home/aegis/igor/LM4paper/tests/testword.txt")
