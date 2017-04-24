# -*- coding: utf-8 -*-
# Project : LM4paper
# Created by igor on 2016/11/29

import math
import tensorflow as tf
from tensorlayer.layers import Layer


def linear(x, size, name):
    w = tf.get_variable(name + "/W", [x.get_shape(-1), size])
    b = tf.get_variable(name + "/b", [1, size],
                        initializer=tf.zeros_initializer)
    return tf.matmul(x, w) + b


def sharded_variable(name, shape, num_shards, dtype=tf.float32, transposed=False):
    '''分片操作'''
    shard_size = int((shape[0] + num_shards - 1) / num_shards)
    if transposed:
        initializer = tf.uniform_unit_scaling_initializer(
            dtype=dtype, )
    else:
        initializer = tf.uniform_unit_scaling_initializer(dtype=dtype, )
    return [tf.get_variable(name + '_%d' % i, [shard_size,
                                               shape[1]],
                            initializer=initializer, dtype=dtype)
            for i in range(num_shards)]


def _get_sharded_variable(name, shape, dtype, num_shards):
    """Get a list of sharded variables with the given dtype."""
    if num_shards > shape[0]:
        raise ValueError("Too many shards: shape=%s, num_shards=%d" %
                         (shape, num_shards))
    unit_shard_size = int(math.floor(shape[0] / num_shards))
    remaining_rows = shape[0] - unit_shard_size * num_shards

    shards = []
    for i in range(num_shards):
        current_size = unit_shard_size
        if i < remaining_rows:
            current_size += 1
        shards.append(tf.get_variable(name + "_%d" % i, [current_size] + shape[1:],
                                      dtype=dtype))
    return shards


def _get_concat_variable(name, shape, dtype, num_shards):
    """Get a sharded variable concatenated into one tensor"""
    _sharded_variable = _get_sharded_variable(name, shape, dtype, num_shards)
    if len(_sharded_variable) == 1:
        return _sharded_variable[0]
    return tf.concat(0, _sharded_variable)


class LSTMCell(tf.nn.rnn_cell.RNNCell):
    def __init__(self, num_units, input_size, initializer=None,
                 num_proj=None, num_shards=1, dtype=tf.float32):
        self._num_units = num_units
        self._initializer = initializer
        self._num_proj = num_proj
        self._num_units_shards = num_shards
        self._num_proj_shards = num_shards
        self._forget_bias = 1.0

        if num_proj:
            self._state_size = num_units + num_proj
            self._output_size = num_proj
        else:
            self._state_size = 2 * num_units
            self._output_size = num_units

        with tf.variable_scope("LSTMCell"):
            # shape = (input_size + num_proj, 4 * num_units)
            self._concat_w = _get_concat_variable(
                "w", [input_size + num_proj, 4 * self._num_units],
                dtype, self._num_units_shards)

            self._b = tf.get_variable(
                "b", shape=[4 * self._num_units],
                initializer=tf.zeros_initializer, dtype=dtype)

            self._concat_w_proj = _get_concat_variable(
                "W_P", [self._num_units, self._num_proj],
                dtype, self._num_proj_shards)

    @property
    def state_size(self):
        return self._state_size

    @property
    def output_size(self):
        return self._output_size

    def __call__(self, inputs, state, scope=None):
        num_proj = self._num_units if self._num_proj is None else self._num_proj

        c_prev = tf.slice(state, [0, 0], [-1, self._num_units])
        m_prev = tf.slice(state, [0, self._num_units], [-1, num_proj])

        input_size = inputs.get_shape().with_rank(2)[1]
        if input_size.value is None:
            raise ValueError("Could not infer input size from inputs.get_shape()[-1]")

        with tf.variable_scope(type(self).__name__,
                               initializer=self._initializer):
            # i = input_gate, j = new_input, f = forget_gate, o = output_gate
            cell_inputs = tf.concat(1, [inputs, m_prev])
            lstm_matrix = tf.nn.bias_add(tf.matmul(cell_inputs, self._concat_w),
                                         self._b)
            i, j, f, o = tf.split(1, 4, lstm_matrix)

            c = tf.sigmoid(f + self._forget_bias) * c_prev + tf.sigmoid(i) * tf.sigmoid(j)

            m = tf.sigmoid(o) * tf.tanh(c)

            if self._num_proj is not None:
                m = tf.matmul(m, self._concat_w_proj)

        new_state = tf.concat(1, [c, m])
        return m, new_state


class EmbeddingInputlayer(Layer):
    """
    The :class:`EmbeddingInputlayer` class is a fully connected layer,
    for Word Embedding. Words are input as integer index.
    The output is the embedded word vector.

    If you have a pre-train matrix, you can assign the matrix into it.
    To train a word embedding matrix, you can used class:`Word2vecEmbeddingInputlayer`.

    Note that, do not update this embedding matrix.

    Parameters
    ----------
    inputs : placeholder
        For word inputs. integer index format.
        a 2D tensor : [batch_size, num_steps(num_words)]
    vocabulary_size : int
        The size of vocabulary, number of words.
    embedding_size : int
        The number of embedding dimensions.
    E_init : embedding initializer
        The initializer for initializing the embedding matrix.
    E_init_args : a dictionary
        The arguments for embedding initializer
    name : a string or None
        An optional name to attach to this layer.

    Variables
    ------------
    outputs : a tensor
        The outputs of embedding layer.
        the outputs 3D tensor : [batch_size, num_steps(num_words), embedding_size]


    """

    def __init__(
            self,
            inputs=None,
            vocabulary_size=80000,
            embedding_size=200,
            num_shards=None,
            E_init=tf.random_uniform_initializer(-0.1, 0.1),
            E_init_args={},
            name='embedding_layer',
    ):
        Layer.__init__(self, name=name)
        self.inputs = inputs
        print(
            "  tensorlayer:Instantiate EmbeddingInputlayer %s: (%d, %d)" % (self.name, vocabulary_size, embedding_size))

        with tf.variable_scope(name) as vs:
            embeddings = sharded_variable("emb", [vocabulary_size, embedding_size], num_shards)
            embed = tf.nn.embedding_lookup(embeddings, self.inputs)

        self.outputs = embed

        self.all_layers = [self.outputs]
        self.all_params = [embeddings]
        self.all_drop = {}


class CLstmProjectLayer(Layer):
    """
    The :class:`RNNLayer` class is a CLSTM with project layer
    Parameters
    ----------
    layer : a :class:`Layer` instance
        The `Layer` class feeding into this layer.
    cell_fn : a TensorFlow's core RNN cell as follow.
        - see `RNN Cells in TensorFlow <https://www.tensorflow.org/versions/master/api_docs/python/rnn_cell.html>`_
        - class ``tf.nn.rnn_cell.LSTMCell``
    cell_init_args : a dictionary
        The arguments for the cell initializer.
    n_hidden : a int
        The number of hidden units in the layer.
    initializer : initializer
        The initializer for initializing the parameters.
    n_steps : a int
        The sequence length.
    initial_state : None or RNN State
        If None, initial_state is zero_state.
    return_last : boolean
        - If True, return the last output, "Sequence input and single output"
        - If False, return all outputs, "Synced sequence input and output"
        - In other word, if you want to apply one or more RNN(s) on this layer, set to False.
    return_seq_2d : boolean
        - When return_last = False
        - If True, return 2D Tensor [n_example, n_hidden], for stacking DenseLayer after it.
        - If False, return 3D Tensor [n_example/n_steps, n_steps, n_hidden], for stacking multiple RNN after it.
    name : a string or None
        An optional name to attach to this layer.

    Variables
    --------------
    outputs : a tensor
        The output of this RNN.
        return_last = False, outputs = all cell_output, which is the hidden state.
            cell_output.get_shape() = (?, n_hidden)

    final_state : a tensor or StateTuple
        When state_is_tuple = False,
        it is the final hidden and cell states, states.get_shape() = [?, 2 * n_hidden].\n
        When state_is_tuple = True, it stores two elements: (c, h), in that order.
        You can get the final state after each iteration during training, then
        feed it to the initial state of next iteration.

    initial_state : a tensor or StateTuple
        It is the initial state of this RNN layer, you can use it to initialize
        your state at the begining of each epoch or iteration according to your
        training procedure.

    batch_size : int or tensor
        Is int, if able to compute the batch_size, otherwise, tensor for ``?``.


    Notes
    -----
    Input dimension should be rank 3 : [batch_size, n_steps, n_features], if no, please see :class:`ReshapeLayer`.

    """

    def __init__(
            self,
            cnn_inputs,
            cnn_size,
            input_size,
            layer=None,
            n_hidden=100,
            num_proj=None,
            cell_init_args={},
            initializer=tf.random_uniform_initializer(-0.1, 0.1),
            n_steps=5,
            initial_state=None,
            return_last=False,
            # is_reshape = True,
            return_seq_2d=False,
            name='lstm_layer',
    ):
        Layer.__init__(self, name=name)
        self.inputs = layer.outputs

        cell_fn = CLSTMCell

        print("  tensorlayer:Instantiate RNNLayer %s: n_hidden:%d, n_steps:%d, in_dim:%d %s, cell_fn:%s " % (
            self.name, n_hidden,
            n_steps, self.inputs.get_shape().ndims, self.inputs.get_shape(), cell_fn.__name__))
        # You can get the dimension by .get_shape() or ._shape, and check the
        # dimension by .with_rank() as follow.
        # self.inputs.get_shape().with_rank(2)
        # self.inputs.get_shape().with_rank(3)

        # Input dimension should be rank 3 [batch_size, n_steps(max), n_features]
        try:
            self.inputs.get_shape().with_rank(3)
        except:
            raise Exception("RNN : Input dimension should be rank 3 : [batch_size, n_steps, n_features]")

        # is_reshape : boolean (deprecate)
        #     Reshape the inputs to 3 dimension tensor.\n
        #     If input is［batch_size, n_steps, n_features], we do not need to reshape it.\n
        #     If input is [batch_size * n_steps, n_features], we need to reshape it.
        # if is_reshape:
        #     self.inputs = tf.reshape(self.inputs, shape=[-1, n_steps, int(self.inputs._shape[-1])])

        fixed_batch_size = self.inputs.get_shape().with_rank_at_least(1)[0]

        if fixed_batch_size.value:
            batch_size = fixed_batch_size.value
            print("     RNN batch_size (concurrent processes): %d" % batch_size)
        else:
            from tensorflow.python.ops import array_ops
            batch_size = array_ops.shape(self.inputs)[0]
            print("     non specified batch_size, uses a tensor instead.")
        self.batch_size = batch_size

        # Simplified version of tensorflow.models.rnn.rnn.py's rnn().
        # This builds an unrolled LSTM for tutorial purposes only.
        # In general, use the rnn() or state_saving_rnn() from rnn.py.
        #
        # The alternative version of the code below is:
        #
        # from tensorflow.models.rnn import rnn
        # inputs = [tf.squeeze(input_, [1])
        #           for input_ in tf.split(1, num_steps, inputs)]
        # outputs, state = rnn.rnn(cell, inputs, initial_state=self._initial_state)
        outputs = []

        self.cell = cell = cell_fn(num_units=n_hidden, input_size=input_size, cnn_size=cnn_size, num_proj=num_proj,
                                   **cell_init_args)
        if initial_state is None:
            self.initial_state = cell.zero_state(batch_size, dtype=tf.float32)  # 1.2.3

        state = self.initial_state
        # with tf.variable_scope("model", reuse=None, initializer=initializer):
        with tf.variable_scope(name, initializer=initializer) as vs:
            for time_step in range(n_steps):
                if time_step > 0: tf.get_variable_scope().reuse_variables()
                (cell_output, state) = cell(self.inputs[:, time_step, :], cnn_inputs[:, time_step, :], state)
                outputs.append(cell_output)

            # Retrieve just the RNN variables.
            # rnn_variables = [v for v in tf.all_variables() if v.name.startswith(vs.name)]
            rnn_variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=vs.name)

        print("     n_params : %d" % (len(rnn_variables)))

        output_size = num_proj if num_proj else n_hidden

        if return_last:
            # 2D Tensor [batch_size, n_hidden]
            self.outputs = outputs[-1]
        else:
            if return_seq_2d:
                # PTB tutorial: stack dense layer after that, or compute the cost from the output
                # 2D Tensor [n_example, n_hidden]
                self.outputs = tf.reshape(tf.concat(1, outputs), [-1, output_size])
            else:
                # <akara>: stack more RNN layer after that
                # 3D Tensor [n_example/n_steps, n_steps, n_hidden]
                self.outputs = tf.reshape(tf.concat(1, outputs), [-1, n_steps, output_size])

        self.final_state = state

        self.all_layers = list(layer.all_layers)
        self.all_params = list(layer.all_params)
        self.all_drop = dict(layer.all_drop)
        # print(type(self.outputs))
        self.all_layers.extend([self.outputs])
        self.all_params.extend(rnn_variables)


class CLSTMCell(tf.nn.rnn_cell.RNNCell):
    def __init__(self, num_units, input_size, cnn_size, initializer=None,
                 num_proj=None, num_shards=1, dtype=tf.float32):

        self._num_units = num_units
        self._initializer = initializer
        self._num_proj = num_proj
        self._num_units_shards = num_shards
        self._num_proj_shards = num_shards
        self._forget_bias = 1.0

        if num_proj:
            self._state_size = num_units + num_proj
            self._output_size = num_proj
        else:
            self._state_size = 2 * num_units
            self._output_size = num_units

        with tf.variable_scope("LSTMCell"):
            # shape = (input_size + num_proj, 4 * num_units)
            self._concat_w = _get_concat_variable(
                "w", [input_size + num_proj, 4 * self._num_units],
                dtype, self._num_units_shards)

            self._b = tf.get_variable(
                "b", shape=[4 * self._num_units],
                initializer=tf.zeros_initializer, dtype=dtype)

            self._concat_w_proj = _get_concat_variable(
                "W_P", [self._num_units, self._num_proj],
                dtype, self._num_proj_shards)

            self._w_char = _get_concat_variable(
                "w_char", [cnn_size, input_size],
                dtype, self._num_units_shards)

            self._b_char = tf.get_variable(
                "b_char", shape=[input_size],
                initializer=tf.zeros_initializer, dtype=dtype)

    @property
    def state_size(self):
        return self._state_size

    @property
    def output_size(self):
        return self._output_size

    def __call__(self, inputs, cnn_inputs, state, scope=None):

        num_proj = self._num_units if self._num_proj is None else self._num_proj

        # word-char gate
        cnn_inputs = tf.matmul(cnn_inputs,
                               self._w_char) + self._b_char  # cnn_input projection size to (batch_size,input_size)
        g_w = tf.sigmoid(cnn_inputs)

        inputs = (1 - g_w) * inputs + g_w * inputs

        c_prev = tf.slice(state, [0, 0], [-1, self._num_units])
        m_prev = tf.slice(state, [0, self._num_units], [-1, num_proj])

        input_size = inputs.get_shape().with_rank(2)[1]
        if input_size.value is None:
            raise ValueError("Could not infer input size from inputs.get_shape()[-1]")

        with tf.variable_scope(type(self).__name__,
                               initializer=self._initializer):
            # i = input_gate, j = new_input, f = forget_gate, o = output_gate
            cell_inputs = tf.concat(1, [inputs, m_prev])
            lstm_matrix = tf.nn.bias_add(tf.matmul(cell_inputs, self._concat_w),
                                         self._b)
            i, j, f, o = tf.split(1, 4, lstm_matrix)

            c = tf.sigmoid(f + self._forget_bias) * c_prev + tf.sigmoid(i) * tf.sigmoid(j)

            m = tf.sigmoid(o) * tf.tanh(c)

            if self._num_proj is not None:
                m = tf.matmul(m, self._concat_w_proj)

        new_state = tf.concat(1, [c, m])
        return m, new_state


if __name__ == '__main__':
    cell = CLSTMCell(num_units=100, input_size=100, cnn_size=30, num_proj=512)
    x = tf.placeholder(tf.float32, shape=(60, 100))
    y = tf.placeholder(tf.float32, shape=(60, 30))
    cell(x, y, None)
