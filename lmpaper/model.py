#!/usr/bin/env python
# -*- coding: utf-8 -*-
# author igor
# Created by iFantastic on 16-5-5
'''
language model 模型
'''

import re

import tensorflow as tf

from lmpaper.context import TrainConfig, TestConfig

TOWER_NAME = 'tower'
MOVING_AVERAGE_DECAY = 0.9999  # 移动平均衰减


def _variable_on_cpu(name, shape, initializer):
    '''
    辅助函数:在CPU中创建Variable
    :param name:变量名称
    :param shape:形状
    :param initializer:初始化
    :return:
        Variable Tensor
    '''
    with tf.device('/cpu:0'):
        var = tf.get_variable(name, shape, initializer=initializer)
    return var


def _variable_with_weight_decay(name, shape, stddev, wd):
    '''
    辅助函数:利用权重衰减初始化Variable

    变量利用truncated normal distribution
    只有指定的时候才进行权重衰减(weight decay)
    :param name:变量的名称
    :param shape:形状
    :param stddev:trucated Gaussian的标准差
    :param wd:增加 L2Loss 权重衰减,如果是None,weight decay不被增加

    :return:
        Variabel Tensor
    '''
    var = _variable_on_cpu(name, shape,
                           tf.truncated_normal_initializer(stddev=stddev))
    if wd:
        weight_decay = tf.mul(tf.nn.l2_loss(var), wd, name='weight_loss')
        # Stores value in the collection with the given name
        tf.add_to_collection('losses', weight_decay)
    return var


def _activation_summary(x):
    '''
    辅助函数:创建激活神经元的summaries

    提供激活元的直方图的总结
    提供激活元稀疏测量的总结

    :param x:Tensor
    :return:
        nothing
    '''
    tensor_name = re.sub('%s_[0-9]*/' % TOWER_NAME, '', x.op.name)
    tf.histogram_summary(tensor_name + '/activations', x)
    # 0的占比,稀疏性
    tf.scalar_summary(tensor_name + '/sparsity', tf.nn.zero_fraction(x))


def _add_loss_summaries(total_loss):
    '''
    添加损失总结
    生成所有损失和相关总结的移动平均:为可视化网络的性能
    :param total_loss:Total loss from loss()
    :return:
        loss_averages_op:op for generating moving averages of losses.
    '''

    # 为单个的损失和总共的损失计算指数移动平均
    loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
    losses = tf.get_collection('losses')
    loss_averages_op = loss_averages.apply(losses + [total_loss])

    for l in losses + [total_loss]:
        tf.scalar_summary(l.op.name + '(raw)', l)
        tf.scalar_summary(l.op.name, loss_averages.average(l))
    return loss_averages_op


def inference_with_buckets(features, buckets=None, is_training=False, config=None):
    if config:
        config = config
    elif is_training:
        # 训练的超参数
        config = TrainConfig()
    else:
        config = TestConfig()

    assert config is not None, "config can't be None"

    outputs = []
    output_states = []
    # Embedding layer
    # 构建词嵌入层

    with tf.device('/cpu:0'):
        with tf.variable_scope('embedding_layer') as scope:
            initializer = tf.random_uniform([config.vocab_size, config.hidden_size],
                                            -1.0, 1.0)
            embedding = _variable_on_cpu('embedding', shape=None,
                                         initializer=initializer)

            lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(config.hidden_size,
                                                     forget_bias=0.0)

    if is_training and config.keep_prob < 1:  # 如果是训练,dropout正则化,防止过拟合
        lstm_cell = tf.nn.rnn_cell.DropoutWrapper(
            lstm_cell, output_keep_prob=config.keep_prob)

    cell = tf.nn.rnn_cell.MultiRNNCell([lstm_cell] * config.num_layers)

    # 初始状态
    initial_state = cell.zero_state(config.batch_size, tf.float32)

    with tf.op_scope(features, None, "model_with_buckets"):
        for j, bucket in enumerate(buckets):
            inputs_j = features[j]
            # 3D Tensor [batch_size,num_steps,hidden_size]
            features_embedding = tf.nn.embedding_lookup(embedding, inputs_j)

            if is_training and config.keep_prob < 1:
                features_embedding = tf.nn.dropout(features_embedding, config.keep_prob)

            with tf.variable_scope(tf.get_variable_scope(),
                                   reuse=True if j > 0 else None):
                state = initial_state
                features_embedding = [tf.squeeze(_input, [1]) for _input in tf.split(1, bucket - 1,
                                                                                     features_embedding)]
                bucket_outputs, state = tf.nn.rnn(cell, features_embedding, initial_state=state)

                # shape:[batch_size * bucket,hidden_size]
                output_wb = tf.reshape(tf.concat(1, bucket_outputs), shape=[-1, config.hidden_size])

                outputs.append(output_wb)
                output_states.append(state)

                for op in bucket_outputs:
                    _activation_summary(op)

    # 2D Tensor [batch_size*num_steps,hidden_size]
    output = tf.reshape(tf.concat(0, outputs), shape=[-1, config.hidden_size])

    with tf.variable_scope("output") as scope:
        softmax_w = _variable_with_weight_decay('softmax_w',
                                                shape=[config.hidden_size, config.vocab_size],
                                                stddev=1 / config.hidden_size, wd=0.0)
        softmax_b = _variable_on_cpu('softmax_b', [config.vocab_size],
                                     initializer=tf.constant_initializer(0.0))

        # 2D Tensor [sum(batch_size * [buckets]), vocab_size]
        logits = tf.matmul(output, softmax_w) + softmax_b

        _activation_summary(logits)

    return logits, initial_state, output_states


def loss_with_buckets(logits, targets, buckets, config):
    '''
    计算代价
    :param logits: by inference_with_buckets() list of Tensor
    :param targets:预测目标
    :param config:参数
    :return:total_loss
    '''
    buckets = [bk - 1 for bk in buckets]
    targets = tf.concat(0, [tf.reshape(target, [-1]) for target in targets])
    losses = tf.nn.seq2seq.sequence_loss_by_example(
        [logits],
        [targets],
        [tf.ones([config.batch_size * sum(buckets)])]
    )
    costs = tf.reduce_sum(losses) / config.batch_size * len(buckets)
    tf.scalar_summary('costs', costs)
    tf.add_to_collection('losses', costs)
    return tf.add_n(tf.get_collection('losses'), name='total_loss')


def inference(features, is_training=False, config=None):
    '''
    模型推理过程
    :param features:2D Tensor [batch_size,num_steps]
    :param is_training:是否是训练
    :return:logits,final_state
    '''
    if config:
        config = config
    elif is_training:
        # 训练的超参数
        config = TrainConfig()
    else:
        config = TestConfig()

    assert config is not None, "config can't be None"
    lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(config.hidden_size,
                                             forget_bias=0.0)
    if is_training and config.keep_prob < 1:  # 如果是训练,dropout正则化,防止过拟合
        lstm_cell = tf.nn.rnn_cell.DropoutWrapper(
            lstm_cell, output_keep_prob=config.keep_prob)

    cell = tf.nn.rnn_cell.MultiRNNCell([lstm_cell] * config.num_layers)

    # 初始状态
    initial_state = cell.zero_state(config.batch_size, tf.float32)

    # Embedding layer
    with tf.variable_scope('embedding_layer') as scope:
        initializer = tf.random_uniform([config.vocab_size, config.hidden_size],
                                        -1.0, 1.0)
        embedding = _variable_on_cpu('embedding', shape=None,
                                     initializer=initializer)

        # 3D Tensor [batch_size,num_steps,hidden_size]
        features_embedding = tf.nn.embedding_lookup(embedding, features)

    if is_training and config.keep_prob < 1:
        features_embedding = tf.nn.dropout(features_embedding, config.keep_prob)

    # outputs = []
    state = initial_state
    features_embedding = [tf.squeeze(_input, [1]) for _input in tf.split(1, config.num_steps,
                                                                         features_embedding)]
    outputs, state = tf.nn.rnn(cell, features_embedding, initial_state=state)
    # with tf.variable_scope("RNN") as scope:
    #     for time_step in range(config.num_steps):  # 展开
    #         if time_step > 0:
    #             tf.get_variable_scope().reuse_variables()
    #         (cell_output, state) = cell(features_embedding[:, time_step, :], state)
    #         outputs.append(cell_output)

    for op in outputs:
        _activation_summary(op)

    # 2D Tensor [batch_size*num_steps,hidden_size]
    output = tf.reshape(tf.concat(1, outputs), shape=[-1, config.hidden_size])

    with tf.variable_scope("output") as scope:
        softmax_w = _variable_with_weight_decay('softmax_w',
                                                shape=[config.hidden_size, config.vocab_size],
                                                stddev=1 / config.hidden_size, wd=0.0)
        softmax_b = _variable_on_cpu('softmax_b', [config.vocab_size],
                                     initializer=tf.constant_initializer(0.0))

        # 2D Tensor [batch_size * nums_step, vocab_size]
        logits = tf.matmul(output, softmax_w) + softmax_b

        _activation_summary(logits)

    return logits, initial_state, state


def loss(logits, targets, config):
    '''
    计算代价
    :param logits: by inference()
    :param targets:预测目标
    :param config:参数
    :return:total_loss
    '''

    losses = tf.nn.seq2seq.sequence_loss_by_example(
        [logits],
        [tf.reshape(targets, [-1])],
        [tf.ones([config.batch_size * config.num_steps])]
    )
    costs = tf.reduce_sum(losses) / config.batch_size
    tf.scalar_summary('costs', costs)
    tf.add_to_collection('losses', costs)
    return tf.add_n(tf.get_collection('losses'), name='total_loss')


def train(total_loss, learning_rate, global_step):
    '''
    语言模型训练过程
    :param total_loss: by loss()
    :param learning_rate 1D Tensor
    :param global_step:
    :return:
        train_op
    '''

    # tvars = tf.trainable_variables()
    # grads, _ = tf.clip_by_global_norm(tf.gradients(total_loss, tvars),
    #                                   clip_norm=5)
    # optimizer = tf.text.GradientDescentOptimizer(learning_rate)
    # train_op = optimizer.apply_gradients(zip(grads, tvars))

    loss_average_op = _add_loss_summaries(total_loss)

    with tf.control_dependencies([loss_average_op]):
        opt = tf.train.GradientDescentOptimizer(learning_rate)
        grads = opt.compute_gradients(total_loss)

    # Apply gradients
    apply_gradient_op = opt.apply_gradients(grads, global_step)

    for var in tf.trainable_variables():
        tf.histogram_summary(var.op.name, var)

    # 追踪训练变量的指数移动平均
    variable_averages = tf.train.ExponentialMovingAverage(
        MOVING_AVERAGE_DECAY, global_step)
    variable_averages_op = variable_averages.apply(tf.trainable_variables())

    with tf.control_dependencies([apply_gradient_op, variable_averages_op]):
        train_op = tf.no_op(name='train_op')
    return train_op
