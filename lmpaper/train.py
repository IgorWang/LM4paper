#!/usr/bin/env python
# -*- coding: utf-8 -*-
# author igor
# Created by iFantastic on 16-5-5
'''
训练模型
保存模型
'''
import os
import time

import numpy as np
import tensorflow as tf

from lmpaper.context import *
from lmpaper.model import *
from lmpaper.reader import distorted_inputs, inputs_with_buckets, TOTAL_EXAMLES
from lmpaper.proprecess import load_dictionary, random_read, iterator

tf.app.flags.DEFINE_boolean('log_device_placement', False,
                            'where to log device placement.')


def train_model_with_buckets(verbose=False, buckets=None):
    train_path = os.path.join(DATA_DIR, 'train')
    file_with_buckets = [os.path.join(train_path, 'train_%s.csv' % (str(bucket))) for bucket in buckets]
    config = TrainConfig()

    with tf.Graph().as_default(), tf.Session() as sess:
        global_step = tf.Variable(0, trainable=False)

        learning_rate = tf.placeholder(dtype=tf.float32, name='learning_rate')

        feature_wb, targets_wb = inputs_with_buckets(file_with_buckets, buckets, config.batch_size)

        logits_wb, init_state, final_state_wb = inference_with_buckets(feature_wb, buckets, True, config)

        losses = loss_with_buckets(logits_wb, targets_wb, buckets, config)

        train_op = train(losses, learning_rate, global_step)

        saver = tf.train.Saver(tf.all_variables())

        summary_op = tf.merge_all_summaries()

        init = tf.initialize_all_variables().run(session=sess)

        tf.train.start_queue_runners(sess=sess)

        summary_writer = tf.train.SummaryWriter(FLAGS.model_dir,
                                                graph_def=sess.graph_def)
        lr = config.learning_rate
        gs = 0
        for epoch_step in range(config.max_max_epoch):
            lr_decay = config.lr_decay ** max(epoch_step - config.max_epoch, 0.0)
            lr = lr * lr_decay
            print("Epoch : %d with Learning rate: %.3f" % (epoch_step + 1, lr))

            # run a single epoch based on given random data
            epoch_size = 1000
            costs = 0.0
            iters = 0

            state = init_state.eval()
            start_time = time.time()
            for epoch in range(epoch_size):
                # x, y = sess.run([features, targets])
                feed_dict = {learning_rate: lr,
                             init_state: state}
                cost, _ = sess.run([losses, train_op],
                                   feed_dict=feed_dict)
                gs += 1
                costs += cost
                iters += sum(BUCKETS)

                if verbose and epoch % (epoch_size // 10) == 10:
                    print("completeness ： %.3f --- preplexity : %.3f --- speed : %.3f wps" %
                          (epoch * 1.0 / epoch_size, np.exp(costs / iters),
                           (time.time() - start_time)))
                    summary_str = sess.run(summary_op)
                    summary_writer.add_summary(summary_str, gs)
                if gs % 2000 == 0 or (epoch_step + 1) == config.max_max_epoch:
                    checkpoint_path = os.path.join(FLAGS.model_dir, "model.ckpt")
                    saver.save(sess, checkpoint_path, global_step=gs)


def train_model(verbose=False):
    train_path = os.path.join(DATA_DIR, 'train')
    filenames = [os.path.join(train_path, i) for i in os.listdir(train_path)]
    config = TrainConfig()

    with tf.Graph().as_default(), tf.Session() as sess:
        global_step = tf.Variable(0, trainable=False)

        with tf.device('/cpu:0'):
            features, targets = distorted_inputs(filenames, config.batch_size)

        # input_xs = tf.placeholder(dtype=tf.int32, shape=[config.batch_size,
        #                                                  config.num_steps], name='input_words')
        # input_ys = tf.placeholder(dtype=tf.int32, shape=[config.batch_size,
        #                                                  config.num_steps], name='target_words')

        learning_rate = tf.placeholder(dtype=tf.float32, name='learning_rate')

        logits, init_state, final_state = inference(features, True, config)

        losses = loss(logits, targets, config)

        train_op = train(losses, learning_rate, global_step)

        saver = tf.train.Saver(tf.all_variables())

        summary_op = tf.merge_all_summaries()

        init = tf.initialize_all_variables()

        tf.train.start_queue_runners(sess=sess)

        sess.run(init)

        summary_writer = tf.train.SummaryWriter(FLAGS.model_dir,
                                                graph_def=sess.graph_def)
        lr = config.learning_rate
        gs = 0
        for epoch_step in range(config.max_max_epoch):
            lr_decay = config.lr_decay ** max(epoch_step - config.max_epoch, 0.0)
            lr = lr * lr_decay
            print("Epoch : %d with Learning rate: %.3f" % (epoch_step + 1, lr))

            # run a single epoch based on given random data
            epoch_size = 1000
            costs = 0.0
            iters = 0

            state = init_state.eval()
            start_time = time.time()

            for epoch in range(epoch_size):
                # x, y = sess.run([features, targets])
                feed_dict = {learning_rate: lr,
                             init_state: state}
                cost, state, _ = sess.run([losses, final_state, train_op],
                                          feed_dict=feed_dict)
                gs += 1
                costs += cost
                iters += config.num_steps

                if verbose and epoch % (epoch_size // 10) == 10:
                    print("completeness ： %.3f --- preplexity : %.3f --- speed : %.3f wps" %
                          (epoch * 1.0 / epoch_size, np.exp(costs / iters),
                           (time.time() - start_time)))
                    summary_str = sess.run(summary_op)
                    summary_writer.add_summary(summary_str, gs)
                if gs % 2000 == 0 or (epoch_step + 1) == config.max_max_epoch:
                    checkpoint_path = os.path.join(FLAGS.model_dir, "model.ckpt")
                    saver.save(sess, checkpoint_path, global_step=gs)


def old_train_model(verbose=False):
    dictionary_path = os.path.join(FLAGS.model_dir, 'dict.pickle')
    token2id = load_dictionary(dictionary_path)
    raw_data_path = os.path.join(DATA_DIR, 'text', 'text.txt')
    config = TrainConfig()
    config.vocab_size = len(token2id)
    train_data = random_read(config.batch_size, raw_data_path, config.max_max_epoch, multiple=100)

    with tf.Graph().as_default(), tf.Session() as sess:
        global_step = tf.Variable(0, trainable=False)

        features = tf.placeholder(dtype=tf.int32, shape=[config.batch_size,
                                                         config.num_steps], name='input_words')
        targets = tf.placeholder(dtype=tf.int32, shape=[config.batch_size,
                                                        config.num_steps], name='target_words')
        learning_rate = tf.placeholder(dtype=tf.float32, name='learning_rate')

        logits, init_state, final_state = inference(features, True, config)

        losses = loss(logits, targets, config)

        train_op = train(losses, learning_rate, global_step)

        saver = tf.train.Saver(tf.all_variables())

        summary_op = tf.merge_all_summaries()

        init = tf.initialize_all_variables()

        sess.run(init)

        summary_writer = tf.train.SummaryWriter(FLAGS.model_dir,
                                                graph_def=sess.graph_def)
        lr = config.learning_rate
        gs = 0  # gloable_step
        for epoch_step, data in enumerate(train_data):
            lr_decay = config.lr_decay ** max(epoch_step - config.max_epoch, 0.0)
            lr = lr * lr_decay
            print("Epoch : %d with Learning rate: %.3f" % (epoch_step + 1, lr))

            # run a single epoch based on given random data
            epoch_size = ((len(data)) // config.batch_size - 1) // config.num_steps
            costs = 0.0
            iters = 0

            state = init_state.eval()
            start_time = time.time()
            for step, (x, y) in enumerate(iterator(token2id, data,
                                                   config.batch_size, config.num_steps)):
                feed_dict = {features: x,
                             targets: y,
                             learning_rate: lr,
                             init_state: state}
                cost, state, _ = sess.run([losses, final_state, train_op],
                                          feed_dict=feed_dict)

                gs += 1
                costs += cost
                iters += config.num_steps

                if verbose and step % (epoch_size // 10) == 10:
                    print("completeness ： %.3f --- preplexity : %.3f --- speed : %.3f wps" %
                          (step * 1.0 / epoch_size, np.exp(costs / iters),
                           (time.time() - start_time)))
                    summary_str = sess.run(summary_op, feed_dict)
                    summary_writer.add_summary(summary_str, gs)

                if gs % 2000 == 0 or (epoch_step + 1) == config.max_max_epoch:
                    checkpoint_path = os.path.join(FLAGS.model_dir, "model.ckpt")
                    saver.save(sess, checkpoint_path, global_step=gs)


if __name__ == '__main__':
    # train_model(True)
    train_model_with_buckets(True, BUCKETS)
