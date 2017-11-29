#!/usr/bin/env python
#-*- coding:utf-8 -*-

"""
    *.py: Description of what * does.
    Last Modified:
"""

__author__ = "Sathappan Muthiah"
__email__ = "sathap1@vt.edu"
__version__ = "0.0.1"


import tensorflow as tf


class BaseRNNPtProc(object):
    def __init__(self, hidden_size, batch_size, var_initializer=tf.contrib.layers.xavier_initializer,
                 cell_type=tf.contrib.rnn.BasicRNNCell, num_input_vars=1,
                 input_class_size=[1], input_embedding_size=1, clipping_val=None,
                 losstype="intensity"):
        self.state_size = hidden_size
        self.batch_size = batch_size
        self.var_initializer = var_initializer
        self.cell_type = cell_type
        self.num_input_vars = num_input_vars
        self.input_class_size = input_class_size
        self.clipping_val = clipping_val
        self.losstype = losstype

        if not isinstance(input_embedding_size, list):
            self.input_emb_size = [input_embedding_size for i in range(1, num_input_vars)]
            self.input_emb_size.insert(0, 1)
        else:
            self.input_emb_size = input_embedding_size

    def read_inputs(self):
        pass

    def init_variables(self, num_input_vars=1):
        pass

    def calc_loss(self):
        pass

    def build_graph(self):
        pass

    def run(self, **kwargs):
        if 'sess' in kwargs:
            sess = kwargs['sess']
        else:
            sess = tf.Session()

        if not kwargs.get('reuse', False):
            #tf.reset_default_graph()
            self.init_variables()
            self.build_graph(learning_rate=kwargs.get("learning_rate", 1e-4))
            init = tf.global_variables_initializer()
            init.run(session=sess)

        for ep in range(kwargs['epochs']):
            _, cost = sess.run([self._train, self._cost], feed_dict=kwargs['tf_inputs'])
            if ep % 100 == 0:
                print("Iter:{}, cost: {}".format(ep, cost))

        return sess


class RMTP_TIME(BaseRNNPtProc):
    def init_variables(self):
        self._seqlenmask = tf.placeholder(shape=[self.batch_size], dtype=tf.float32, name="lenmask")
        self._inputs = tf.placeholder(shape=[None, None, 1], dtype=tf.float32, name="x_in")
        self._labels = tf.placeholder(shape=[None, None, 1], dtype=tf.float32, name="y_out")

        self._V = tf.get_variable('V', shape=[self.state_size, 1],
                                  initializer=self.var_initializer())

        self._bo = tf.get_variable('bo', shape=[1],
                                   initializer=tf.constant_initializer(0.))

        self._wo = tf.get_variable("wo", shape=[1], dtype=tf.float32,
                                   initializer=self.var_initializer())

    def build_graph(self, learning_rate=1e-4):
        ### Build RNN Cell and get RNN Outputs
        self._rnnunit = self.cell_type(self.state_size, activation=tf.nn.relu)
        self._initstate = self._rnnunit.zero_state(self.batch_size, tf.float32)

        self._rnnoutputs, self._final_states = tf.nn.dynamic_rnn(self._rnnunit, self._inputs,
                                                                 sequence_length=self._seqlenmask,
                                                                 initial_state=self._initstate)

        ### Process RNN output to get model output and also calc loss
        ### i.e., perform output activation
        self._rnnoutputs_flattened = tf.reshape(self._rnnoutputs, [-1, self._rnnoutputs.shape[-1].value])
        self._labels_flattened = tf.reshape(self._labels, [-1, self._labels.shape[-1].value])
        self._timeloss = self.calc_loss(self._labels_flattened, self._rnnoutputs_flattened)
        self._cost = tf.reduce_sum(self._timeloss)

        ### initialize Optimizer
        self._optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        if self.clipping_val:
            grads, tfvars = zip(*self._optimizer.compute_gradients(self._cost))
            capped_gvs = tf.clip_by_global_norm(grads, self.clipping_val)[0]
            self._train = self._optimizer.apply_gradients(zip(capped_gvs, tfvars))
        else:
            self._train = self._optimizer.minimize(self._cost)

    def calc_loss(self, current_time, rnnstates):
        if self.losstype == "intensity":
            print("reached here")
            self._hist_influence = tf.matmul(rnnstates, self._V)
            self._curr_influence = self._wo * current_time
            self._rate_t = self._hist_influence + self._curr_influence + self._bo
            return -(self._rate_t + tf.exp(self._hist_influence + self._bo) * (1 / self._wo)
                     - (1 / self._wo) * tf.exp(self._rate_t))
        elif self.losstype == "mse":
            time_hat = tf.matmul(rnnstates, self._V)
            time_loss = tf.abs(tf.reshape(time_hat, [-1]) - current_time)
            return time_loss


class RMTP(BaseRNNPtProc):
    def __init__(self, hidden_size, batch_size, num_input_vars,
                 input_activation="onehot", **kwargs):
        if num_input_vars < 2:
            raise Exception("Use RMTP Time Instead")

        self.input_activation_type = input_activation
        super(RMTP, self).__init__(hidden_size, batch_size,
                                   num_input_vars=num_input_vars,
                                   **kwargs)

    def init_variables(self):
        self._seqlenmask = tf.placeholder(shape=[self.batch_size], dtype=tf.float32, name="lenmask")
        self._inputs = tf.placeholder(shape=[None, None, self.num_input_vars], dtype=tf.float32, name="x_in")
        self._labels = tf.placeholder(shape=[None, None, self.num_input_vars], dtype=tf.float32, name="y_out")

        ### assume first input variable is time variable (t_j - t_{j-1}), this is continuous
        ### assume all input variables from second are categorical variables for which embedding is to be found
        self._embedding_matrix = {}
        self._inputemb = {}
        tmplist = [self._inputs[:, :, :1]]
        for i in range(1, self.num_input_vars):
            if self.input_activation_type == "embedding":
                self._embedding_matrix[i] = tf.get_variable('inputemb_{}'.format(i),
                                                            shape=[self.input_class_size[i], self.input_emb_size[i]],
                                                            dtype=tf.float32, initializer=self.var_initializer())

                self._inputemb[i] = tf.nn.embedding_lookup(self._embedding_matrix[i],
                                                           tf.cast(self._inputs[:, :, i], dtype=tf.int32),
                                                           name="emb_{}".format(i))
            elif self.input_activation_type == "onehot":
                self._inputemb[i] = tf.one_hot(tf.cast(self._inputs[:, :, i], dtype=tf.int32),
                                               self.input_class_size[i])  # when num_class is large, use tf embedding

            tmplist.append(self._inputemb[i])

        self._rnninputs = tf.concat(tmplist, axis=2)

        self._V = tf.get_variable('V', shape=[self.state_size, 1], dtype=tf.float32,
                                  initializer=self.var_initializer())

        self._bo = tf.get_variable('bo', shape=[1], dtype=tf.float32,
                                   initializer=tf.constant_initializer(0.))

        self._wo = tf.get_variable("wo", shape=[1], dtype=tf.float32,
                                   initializer=tf.constant_initializer(1.0))

    def build_graph(self, learning_rate=1e-4, loss_tradeoff=1):
        ### Build RNN Cell and get RNN Outputs
        self._rnnunit = self.cell_type(self.state_size, activation=tf.nn.relu)
        self._initstate = self._rnnunit.zero_state(self.batch_size, tf.float32)

        self._rnnoutputs, self._final_states = tf.nn.dynamic_rnn(self._rnnunit, self._rnninputs,
                                                                 sequence_length=self._seqlenmask,
                                                                 initial_state=self._initstate)

        ### Calc Hawkes process Likelihood
        self._rnnoutputs_flattened = tf.reshape(self._rnnoutputs, [-1, self._rnnoutputs.shape[-1].value])
        self._time_labels_flattened = tf.reshape(self._labels[:, :, 0], [-1, 1])
        self._timeloss = self.calc_time_loss(self._time_labels_flattened, self._rnnoutputs_flattened)
        self._timeloss = tf.reduce_mean(tf.reduce_sum(self._timeloss, axis=1))

        ### calc marker likelihood
        self._marker_out = {}
        self._loss = {}
        for i in range(1, self.num_input_vars):
            self._marker_out[i], self._loss[i] = self.calc_marker_loss(self._rnnoutputs_flattened,
                                                                       tf.reshape(self._labels[:, :, i], [-1]),
                                                                       self.input_class_size[i],
                                                                       marker_name="softmax-{}".format(i))

        self._markerloss = tf.add_n([mloss for mloss in self._loss.values()])
        self._cost = self._markerloss + loss_tradeoff * self._timeloss

        ### initialize Optimizer
        self._optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        if self.clipping_val:
            grads, tfvars = zip(*self._optimizer.compute_gradients(self._cost))
            capped_gvs = tf.clip_by_global_norm(grads, self.clipping_val)[0]
            self._train = self._optimizer.apply_gradients(zip(capped_gvs, tfvars))
        else:
            self._train = self._optimizer.minimize(self._cost)

    def calc_marker_loss(self, rnnoutputs, marker_labels, marker_size, marker_name):
        mout = tf.layers.dense(rnnoutputs, marker_size, kernel_regularizer=None,
                               name=marker_name, trainable=True, use_bias=True)

        labels_true = tf.one_hot(tf.cast(marker_labels, dtype=tf.int32), marker_size)
        mark_loss = tf.nn.softmax_cross_entropy_with_logits(logits=mout, labels=labels_true)
        #loss = tf.losses.softmax_cross_entropy(onehot_labels=labels_true,
        #                                       logits=mout)

        return mout, tf.reduce_mean(mark_loss)

    def calc_time_loss(self, current_time, rnnstates):
        if self.losstype == "intensity":
            print("reached here")
            self._hist_influence = tf.matmul(rnnstates, self._V)
            self._curr_influence = self._wo * current_time
            self._rate_t = self._hist_influence + self._curr_influence + self._bo
            self._loglik = (self._rate_t + tf.exp(self._hist_influence + self._bo) * (1 / self._wo)
                     - (1 / self._wo) * tf.exp(self._rate_t))
            return - self._loglik
        elif self.losstype == "mse":
            time_hat = tf.matmul(rnnstates, self._V)
            time_loss = tf.abs(tf.reshape(time_hat, [-1]) - current_time)
            return time_loss
