# -*- coding: utf-8 -*-
# Project : LM4paper
# Created by igor on 17-1-11

import tensorflow as tf
import tensorlayer as tl

from bmlm.model_utils import sharded_variable, EmbeddingInputlayer, CLstmProjectLayer
from bmlm.common import assign_to_gpu, average_grads, find_trainable_variables
from bmlm.hparams import HParams


class CLSTMDNN(object):
    '''
    Char CNN + LSTM language model
    '''

    def __init__(self, hps, mode="train", ps_device="/gpu:0"):
        self.hps = hps
        data_size = hps.batch_size * hps.num_gpus

        self.word_x = tf.placeholder(tf.int32, [data_size, hps.num_steps])
        self.char_x = tf.placeholder(tf.int32, [data_size, hps.num_steps, hps.word_length])
        self.y = tf.placeholder(tf.int32, [data_size, hps.num_steps])
        self.w = tf.placeholder(tf.int32, [data_size, hps.num_steps])

        losses = []
        tower_grads = []

        word_inputs = tf.split(0, hps.num_gpus, self.word_x)
        char_inputs = tf.split(0, hps.num_gpus, self.char_x)

        ys = tf.split(0, hps.num_gpus, self.y)
        ws = tf.split(0, hps.num_gpus, self.w)

        # 模型分布 Mutil-tower model
        for i in range(hps.num_gpus):
            with tf.device(assign_to_gpu(i, ps_device)), tf.variable_scope(
                    tf.get_variable_scope(), reuse=True if i > 0 else None):
                tl.layers.set_name_reuse(True)
                loss = self._forward(i, word_inputs[i], char_inputs[i], ys[i], ws[i])
                losses += [loss]
                if mode == "train":
                    cur_grads = self._backward(loss)
                    tower_grads += [cur_grads]
        self.loss = tf.add_n(losses) / len(losses)
        tf.summary.scalar("model/loss", self.loss)
        self.global_step = tf.get_variable("global_step", [], tf.int32,
                                           initializer=tf.zeros_initializer,
                                           trainable=False)

        if mode == "train":
            grads = average_grads(tower_grads)
            optimizer = tf.train.AdagradOptimizer(hps.learning_rate,
                                                  initial_accumulator_value=1.0)
            self.train_op = optimizer.apply_gradients(grads, global_step=self.global_step)
            self.summary_op = tf.summary.merge_all()
        else:
            self.train_op = tf.no_op()

        # 参数的移动平均
        if mode in ["train", "eval"] and hps.average_params:
            with tf.name_scope(None):
                # Keep track of moving average of LSTM variables.
                ema = tf.train.ExponentialMovingAverage(decay=0.999)
                # 找到训练的参数
                variables_to_average = find_trainable_variables("lstm")
                # 执行移动平均
                self.train_op = tf.group(*[self.train_op, ema.apply(variables_to_average)])
                # 存储参数： a map of names to `Variables` to restore.
                self.avg_dict = ema.variables_to_restore(variables_to_average)

    def _forward(self, gpu, word_input, char_input, y, w):
        '''
        :param gpu: number of GPU
        :param word_input: shape is (batch_size,num_steps)
        :param char_input: shape is (batch_size,num_steps,word_length)
        :param y: targets, shape is (batch_size,num_steps)
        :param w: weights

        :return:
        '''

        hps = self.hps
        w = tf.to_float(w)

        # char embedding layer
        char_input = tf.reshape(char_input, [-1, hps.word_length])  # (batch_size*num_steps,word_length)

        self.char_embedding = char_embeddings = tl.layers.EmbeddingInputlayer(
            inputs=char_input, vocabulary_size=hps.char_size,
            embedding_size=hps.emb_char_size, name="emb_char_layer")

        # return shape of (batch_size*num_steps,word_length,emb_size)
        # expand shape, conv2d accept shape is [batch_size,sequence_length,embedding_size,1]
        char_embeddings_expanded = tl.layers.ReshapeLayer(char_embeddings,
                                                          shape=[-1, hps.word_length, hps.emb_char_size, 1])
        # CNN layer
        with tf.variable_scope("Char_CNN_Pool"):
            pooled_outputs = []
            for i, filter_size in enumerate(hps.filter_sizes):
                char_cnn_network = tl.layers.Conv2dLayer(char_embeddings_expanded,
                                                         act=tf.nn.relu,
                                                         shape=[filter_size, hps.emb_char_size, 1, hps.num_filters],
                                                         strides=[1, 1, 1, 1],
                                                         padding="VALID",
                                                         name="conv_%d" % i)
                pool_network = tl.layers.PoolLayer(char_cnn_network,
                                                   ksize=[1, hps.word_length - filter_size + 1, 1, 1],
                                                   strides=[1, 1, 1, 1],
                                                   padding="VALID",
                                                   name="pool_%d" % i)
                pooled_outputs.append(pool_network.outputs)

        # Combine all the pooled features
        num_filters_total = hps.num_filters * len(hps.filter_sizes)  # eg,3*len([2,3,4,5]) = 12
        h_pool = tf.concat(3, pooled_outputs)

        # outputs of char_cnn
        h_pool_flat = tf.reshape(h_pool, [hps.batch_size, hps.num_steps, num_filters_total])

        self.word_embedding = word_embedding = EmbeddingInputlayer(
            inputs=word_input, vocabulary_size=hps.vocab_size, embedding_size=hps.emb_word_size,
            num_shards=hps.num_shards, name="emb_word_layer")
        network = word_embedding

        if hps.keep_prob < 1.0:
            network = tl.layers.DropoutLayer(word_embedding, keep=hps.keep_prob, is_fix=True, name='drop1')

        for i in range(hps.num_layers):
            with tf.device(assign_to_gpu(gpu)), tf.variable_scope("lstm_%d_layer" % i) as vs:
                network = CLstmProjectLayer(
                    cnn_inputs=h_pool_flat,
                    cnn_size=num_filters_total,
                    input_size=hps.emb_word_size,
                    layer=network,
                    n_hidden=hps.state_size,
                    num_proj=hps.projected_size,
                    n_steps=hps.num_steps,
                    return_seq_2d=False if i < (hps.num_layers - 1) else True,
                    name=vs.name)

            if hps.keep_prob < 1.0:
                network = tl.layers.DropoutLayer(network, keep=hps.keep_prob, is_fix=True, name='lstm_drop_%d' % i)

        softmax_w = sharded_variable("softmax_w", [hps.vocab_size, hps.projected_size], hps.num_shards)
        softmax_b = tf.get_variable("softmax_b", [hps.vocab_size])

        if hps.num_sampled == 0:
            full_softmax_w = tf.reshape(tf.concat(1, softmax_w),
                                        [-1, hps.projected_size])
            full_softmax_w = full_softmax_w[:hps.vocab_size, :]

            logits = tf.matmul(network.outputs, full_softmax_w, transpose_b=True) + softmax_b
            self.logits = logits

            targets = tf.reshape(y, [-1])

            loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits, targets)
        else:
            targets = tf.reshape(y, [-1, 1])
            loss = tf.nn.sampled_softmax_loss(softmax_w,
                                              softmax_b,
                                              tf.to_float(network.outputs),
                                              targets,
                                              hps.num_sampled,
                                              hps.vocab_size)
        loss = tf.reduce_mean(loss * tf.reshape(w, [-1]))
        return loss

    def _backward(self, loss, summaries=False):
        hps = self.hps
        loss = loss * hps.num_steps

        emb_vars = find_trainable_variables("emb")
        lstm_vars = find_trainable_variables("LSTM")
        softmax_vars = find_trainable_variables("softmax")

        all_vars = emb_vars + lstm_vars + softmax_vars

        grads = tf.gradients(loss, all_vars)

        orig_grads = grads[:]
        emb_grads = grads[:len(emb_vars)]
        grads = grads[len(emb_vars):]

        for i in range(len(emb_grads)):
            assert isinstance(emb_grads[i], tf.IndexedSlices)
            emb_grads[i] = tf.IndexedSlices(emb_grads[i].values * hps.batch_size,
                                            emb_grads[i].indices,
                                            emb_grads[i].dense_shape)
        lstm_grads = grads[:len(lstm_vars)]
        softmax_grads = grads[len(lstm_vars):]

        # 梯度修剪，对梯度做一个缩放，防止梯度过大造成的梯度爆炸
        lstm_grads, lstm_norm = tf.clip_by_global_norm(lstm_grads,
                                                       hps.max_grad_norm)

        clipped_grads = emb_grads + lstm_grads + softmax_grads

        assert len(clipped_grads) == len(orig_grads)

        if summaries:
            tf.summary.scalar("model/lstm_grad_norm", lstm_norm)
            tf.summary.scalar("model/lstm_grad_scale", tf.minimum(
                hps.max_grad_norm / lstm_norm, 1.0))
            tf.summary.scalar("model/lstm_weight_norm", tf.global_norm(lstm_vars))

        return list(zip(clipped_grads, all_vars))

    @staticmethod
    def get_default_hparams():

        return HParams(
            batch_size=256,
            num_steps=20,
            num_shards=8,
            word_length=16,
            num_layers=3,
            learning_rate=0.5,
            max_grad_norm=10.0,
            num_delayed_steps=150,
            keep_prob=0.8,

            vocab_size=793470,
            char_size=128,
            emb_char_size=32,
            emb_word_size=512,
            state_size=2048,
            projected_size=512,
            num_sampled=8192,
            num_gpus=4,

            # CNN
            filter_sizes=[1, 2, 3, 4, 5, 6, 7, 8],
            num_filters=8,

            average_params=True,
            run_profiler=False, )


if __name__ == '__main__':
    hps = CLSTMDNN.get_default_hparams()
    model = CLSTMDNN(hps)
