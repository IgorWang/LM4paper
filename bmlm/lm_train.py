# -*- coding: utf-8 -*-
# Project : LM4paper
# Created by igor on 2016/12/2

import os

import tensorflow as tf

from utils import Vocabulary, Dataset
from bmlm.lm import LM
from bmlm.run_utils import run_train, run_eval

flags = tf.flags

flags.DEFINE_string("vocabdir", "/data/lmdata/dict/", "Vocabulary directory")
flags.DEFINE_string("datadir", "/data/one-billion-lm", "Data directory.")
flags.DEFINE_string("logdir", "/data/lmdata/logdir", "Logging directory.")
flags.DEFINE_string("mode", "eval_", "Whether to run 'train' or 'eval' model.")
flags.DEFINE_string("hpconfig", "", "Overrides default hyper-parameters.")
flags.DEFINE_integer("num_gpus", 1, "Number of GPUs used.")
flags.DEFINE_integer("eval_steps", 70, "Number of eval steps.")

FLAGS = flags.FLAGS


def main(_):
    hps = LM.get_default_hparams().parse(FLAGS.hpconfig)
    hps.num_gpus = FLAGS.num_gpus

    vocab = Vocabulary.from_file(os.path.join(FLAGS.vocabdir, "1b_word_vocab.txt"))

    if FLAGS.mode == "train":
        hps.batch_size = 256
        dataset = Dataset(vocab, FLAGS.datadir + "/training-monolingual.tokenized.shuffled/*")
        run_train(dataset, hps, FLAGS.logdir + "/train", ps_device="/gpu:0")

    elif FLAGS.mode.startswith("eval_"):
        hps.batch_size = 64
        if FLAGS.mode.startswith("eval_train"):
            data_dir = FLAGS.datadir + "/training-monolingual.tokenized.shuffled/*"
        else:
            data_dir = FLAGS.datadir + "/heldout-monolingual.tokenized.shuffled/news.en.heldout-00000-of-00050"
        dataset = Dataset(vocab, data_dir, deterministic=True)
        run_eval(dataset, hps, FLAGS.logdir, FLAGS.mode, FLAGS.eval_steps)


if __name__ == "__main__":
    tf.app.run()
