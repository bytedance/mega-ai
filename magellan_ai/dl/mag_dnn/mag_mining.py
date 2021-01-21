# coding: utf-8
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import roc_auc_score
from time import time

import numpy as np
import pandas as pd
import tensorflow._api.v2.compat.v1 as tf


class DeepFM(BaseEstimator, TransformerMixin):

    def __init__(self, feature_size, field_size, embedding_size=8,
                 use_fm=True, use_deep=True, l2_reg=0.0,
                 dropout_fm=[1, 1], dropout_deep=[1, 1, 1],
                 deep_hidden_layers=[32, 32],
                 deep_hidden_activation=tf.nn.relu,
                 epoch=10, batch_size=256, learning_rate=0.01,
                 loss_type="log_loss", optimizer="adam",
                 eval_metric=roc_auc_score, random_seed=2016,
                 verbose=0, bigger_is_better=True):

        assert use_fm or use_deep, "deep or fm framework must be used"
        assert loss_type in ["log_loss", "mse"], \
            "loss_type can be either 'log_loss' " \
            "for classification task or 'mse' for regression task"

        # input parameter
        self.embedding_size = embedding_size
        self.feature_size = feature_size
        self.field_size = field_size

        # deep parameter
        self.deep_layers_activation = deep_hidden_activation
        self.deep_hidden_layers = deep_hidden_layers
        self.dropout_deep = dropout_deep
        self.use_deep = use_deep

        # fm parameter
        self.dropout_fm = dropout_fm
        self.use_fm = use_fm

        # train parameter
        self.learning_rate = learning_rate
        self.optimizer_type = optimizer
        self.random_seed = random_seed
        self.batch_size = batch_size
        self.loss_type = loss_type
        self.verbose = verbose
        self.l2_reg = l2_reg
        self.epoch = epoch

        self.bigger_is_better = bigger_is_better
        self.eval_metric = eval_metric

        self.train_result, self.valid_result = [], []
        self._param = dict()
        self._init_graph()

    def _init_graph(self):
        self.graph = tf.Graph()
        with self.graph.as_default():
            tf.set_random_seed(self.random_seed)
            self.feat_value = tf.placeholder(
                tf.float32, shape=[None, None], name='feat_value')
            self.feat_index = tf.placeholder(
                tf.int32, shape=[None, None], name='feat_index')
            self.label = tf.placeholder(
                tf.float32, shape=[None, 1], name='label')
            self.dropout_keep_fm = tf.placeholder(
                tf.float32, shape=[None], name='dropout_keep_fm')
            self.dropout_keep_deep = tf.placeholder(
                tf.float32, shape=[None], name='dropout_deep_deep')

            self.weights = self._init_weights()

            # fm component
            embeddings_tmp = tf.nn.embedding_lookup(
                self.weights['embedding_matrix'], self.feat_index)

            # shape = sample_num * field_size * 1
            feat_value_tmp = tf.reshape(
                self.feat_value, shape=[-1, self.field_size, 1])

            # e = [[e1, e2, ... em] ^ T ... ]
            # shape = sample_num * field_size * embedding_size
            self.embedding_vectors = tf.multiply(
                embeddings_tmp, feat_value_tmp)

            # first order term
            fm_first_order_weight = tf.nn.embedding_lookup(
                self.weights['one_order_weight_matrix'], self.feat_index)

            self.fm_first_order_vectors = tf.reduce_sum(
                tf.multiply(fm_first_order_weight, feat_value_tmp), 2)
            self.fm_first_order_vectors = tf.nn.dropout(
                x=self.fm_first_order_vectors,
                keep_prob=self.dropout_keep_fm[0])

            # second order term
            # shape = sample_num * embedding_size
            self.summed_features_emb = tf.reduce_sum(self.embedding_vectors, 1)
            self.summed_features_emb_square = tf.square(
                self.summed_features_emb)
            self.squared_features_emb = tf.square(self.embedding_vectors)
            self.squared_sum_features_emb = tf.reduce_sum(
                self.squared_features_emb, 1)
            self.fm_second_order_vectors = 0.5 * tf.subtract(
                self.summed_features_emb_square, self.squared_sum_features_emb)
            self.fm_second_order_vectors = tf.nn.dropout(
                x=self.fm_second_order_vectors,
                keep_prob=self.dropout_keep_fm[1])

            # deep component
            self.y_deep = tf.reshape(
                self.embedding_vectors,
                shape=[-1, self.field_size * self.embedding_size])

            self.y_deep = tf.nn.dropout(
                self.y_deep, self.dropout_keep_deep[0])

            for i in range(0, len(self.deep_hidden_layers)):
                self.y_deep = tf.add(tf.matmul(
                    self.y_deep, self.weights["weight_%d" % i]),
                    self.weights["bias_%d" % i])
                self.y_deep = self.deep_layers_activation(self.y_deep)
                self.y_deep = tf.nn.dropout(
                    x=self.y_deep, keep_prob=self.dropout_keep_deep[i+1])

            # output
            concat_input = 0
            if self.use_fm and self.use_deep:
                concat_input = tf.concat([self.fm_first_order_vectors,
                                          self.fm_second_order_vectors,
                                          self.y_deep], axis=1)
            elif self.use_fm:
                concat_input = tf.concat(
                    [self.fm_first_order_vectors,
                     self.fm_second_order_vectors], axis=1)
            elif self.use_deep:
                concat_input = self.y_deep

            self.output = tf.add(tf.matmul(
                concat_input, self.weights['concat_weight']),
                self.weights['concat_bias'])

            # loss
            if self.loss_type == "log_loss":
                self.out = tf.nn.sigmoid(self.output)
                self.loss = tf.losses.log_loss(self.label, self.output)

            elif self.loss_type == "mse":
                self.loss = tf.nn.l2_loss(tf.subtract(self.label, self.output))

            # l2 regularization on weights l2_loss = sum * 1/2
            if self.l2_reg > 0:
                self.loss += tf.nn.l2_loss(
                    self.weights["concat_weight"],
                    name="hn_l2_loss_init")*2**0.5*0.5*self.l2_reg
                if self.use_deep:
                    for i in range(len(self.deep_hidden_layers)):
                        self.loss += tf.nn.l2_loss(
                            self.weights["weight_%d" % i],
                            name="hn_l2_loss_%d" % i)*2**0.5*0.5*self.l2_reg

            # optimizer
            if self.optimizer_type == "adam":
                self.optimizer = tf.train.AdamOptimizer(
                    learning_rate=self.learning_rate,
                    beta1=0.9, beta2=0.999, epsilon=1e-8).minimize(self.loss)
            elif self.optimizer_type == "ada_grad":
                self.optimizer = tf.train.AdagradOptimizer(
                    learning_rate=self.learning_rate,
                    initial_accumulator_value=1e-8).minimize(self.loss)
            elif self.optimizer_type == "gd":
                self.optimizer = tf.train.GradientDescentOptimizer(
                    learning_rate=self.learning_rate).minimize(self.loss)
            elif self.optimizer_type == "momentum":
                self.optimizer = tf.train.MomentumOptimizer(
                    learning_rate=self.learning_rate, momentum=0.95).minimize(
                    self.loss)

            # create session
            self.saver = tf.train.Saver()
            init = tf.global_variables_initializer()
            self.sess = tf.Session()
            self.writer = tf.summary.FileWriter(
                "../../../data/hn_logs/", self.sess.graph)
            self.sess.run(init)
            # self.saver.save(self.sess, "../../../model/deepFM")

            total_parameters = 0
            for variable in self.weights.values():
                shape = variable.get_shape()
                variable_parameters = 1
                for dim in shape:
                    variable_parameters *= dim
                total_parameters += variable_parameters
            if self.verbose > 0:
                print("#Total number of parameters to be trained: %d"
                      % total_parameters)

    def _init_weights(self):

        # basic layer
        weights = dict()
        weights['embedding_matrix'] = tf.Variable(tf.random_normal(
            shape=[self.feature_size, self.embedding_size],
            mean=0.0, stddev=0.01),
            name='embedding_matrix')
        weights['one_order_weight_matrix'] = tf.Variable(tf.random_normal(
            shape=[self.feature_size, 1],
            mean=0.0, stddev=0.01),
            name='one_order_weight_matrix')

        # deep layer
        num_hidden_layer = len(self.deep_hidden_layers)
        input_size = self.field_size * self.embedding_size
        deep_mean = 0
        deep_std = np.sqrt(2.0 / (input_size + self.deep_hidden_layers[0]))
        weights["weight_0"] = tf.Variable(np.random.normal(
            loc=deep_mean, scale=deep_std,
            size=(input_size, self.deep_hidden_layers[0])),
            dtype=np.float32,
            name='layer_0_weight')
        weights["bias_0"] = tf.Variable(np.random.normal(
            loc=deep_mean, scale=deep_std,
            size=(1, self.deep_hidden_layers[0])),
            dtype=np.float32,
            name='layer_0_bias')
        for i in range(1, num_hidden_layer):
            deep_std = np.sqrt(2.0 / (
                    self.deep_hidden_layers[i - 1] +
                    self.deep_hidden_layers[i]))
            weights["weight_%d" % i] = tf.Variable(np.random.normal(
                loc=deep_mean, scale=deep_std,
                size=(self.deep_hidden_layers[i - 1],
                      self.deep_hidden_layers[i])),
                dtype=np.float32,
                name='layer_%d_weight' % i)
            weights["bias_%d" % i] = tf.Variable(np.random.normal(
                loc=deep_mean, scale=deep_std,
                size=(1, self.deep_hidden_layers[i])),
                dtype=np.float32,
                name='layer_%d_bias' % i)

        # final layer
        concat_input_size = 0
        if self.use_fm and self.use_deep:
            concat_input_size = self.field_size + \
                                self.embedding_size + \
                                self.deep_hidden_layers[-1]
        elif self.use_fm:
            concat_input_size = self.field_size + self.embedding_size
        elif self.use_deep:
            concat_input_size = self.deep_hidden_layers[-1]

        concat_mean, concat_std = 0, np.sqrt(2.0 / (concat_input_size + 1))
        weights['concat_weight'] = tf.Variable(np.random.normal(
            loc=concat_mean, scale=concat_std, size=(concat_input_size, 1)),
                                               dtype=np.float32,
                                               name="concat_weight")
        weights['concat_bias'] = tf.Variable(
            tf.constant(0.01), dtype=np.float32, name="concat_bias")
        return weights

    @staticmethod
    def unified_shuffle(x_index_arr, x_value_arr, y_arr):
        rng_state = np.random.get_state()
        np.random.shuffle(x_index_arr)
        np.random.set_state(rng_state)
        np.random.shuffle(x_value_arr)
        np.random.set_state(rng_state)
        np.random.shuffle(y_arr)

    @staticmethod
    def get_batch(x_index, x_value, y, batch_size, index):
        start = index * batch_size
        end = (index + 1) * batch_size
        end = end if end < len(y) else len(y)
        return x_index[start:end], x_value[start:end], \
            [[y_val] for y_val in y[start:end]]

    def fit_on_batch(self, x_index, x_value, y):

        feed_dict = {
            self.feat_index: np.reshape(x_index, (-1, self.field_size)),
            self.feat_value: np.reshape(x_value, (-1, self.field_size)),
            self.label: np.reshape(y, (-1, 1)),
            self.dropout_keep_deep: self.dropout_deep,
            self.dropout_keep_fm: self.dropout_fm}

        loss, opt = self.sess.run(
            [self.loss, self.optimizer], feed_dict=feed_dict)

        return loss

    def fit(self, x_train_index, x_train_value, y_train, x_valid_index=None,
            x_valid_value=None, y_valid=None, early_stopping=False,
            refit=False, refit_epochs=100, refit_verbose=0,
            stop_threshold=5):

        has_valid = x_valid_index is not None
        x_train_index_arr, x_train_value_arr, y_train_arr = \
            x_train_index.values, x_train_value.values, y_train.values

        for epoch in range(self.epoch):
            t1 = time()
            self.unified_shuffle(
                x_train_index_arr, x_train_value_arr, y_train_arr)
            total_batch = int(len(y_train) / self.batch_size)

            for i in range(total_batch):
                x_index_batch, x_value_batch, y_batch = self.get_batch(
                    x_train_index_arr, x_train_value_arr,
                    y_train_arr, self.batch_size, i)
                self.fit_on_batch(x_index_batch, x_value_batch, y_batch)

            y_pred = self.predict(x_train_index_arr, x_train_value_arr)
            train_result = self.eval_metric(y_train, np.reshape(y_pred, (-1,)))
            self.train_result.append(train_result)

            valid_result = 0
            if has_valid:
                x_valid_index_arr, x_valid_value_arr = \
                    x_valid_index.values, x_valid_value.values
                y_valid_pred = self.predict(
                    x_valid_index_arr, x_valid_value_arr)
                valid_result = self.eval_metric(y_valid, y_valid_pred)
                self.valid_result.append(valid_result)

            if self.verbose > 0 and epoch % self.verbose == 0:
                if has_valid:
                    print("[epoch = %d] train-result=%.4f, "
                          "valid-result=%.4f, training duration=[%.1f s]"
                          % (epoch + 1, train_result,
                             valid_result, time() - t1))
                else:
                    print("[epoch = %d] train-result=%.4f, "
                          "training duration=[%.1f s]"
                          % (epoch + 1, train_result, time() - t1))

            if has_valid and early_stopping and \
                    self.training_termination(
                        self.valid_result, stop_threshold):
                break

        if has_valid and refit:
            if self.bigger_is_better:
                best_valid_score = max(self.valid_result)
            else:
                best_valid_score = min(self.valid_result)
            best_epoch = self.valid_result.index(best_valid_score)
            best_train_score = self.train_result[best_epoch]
            x_index_combine_arr = pd.concat(
                [x_train_index, x_valid_index], axis=0).\
                reset_index(drop=True).values
            x_value_combine_arr = pd.concat(
                [x_train_value, x_valid_value], axis=0).\
                reset_index(drop=True).values
            y_combine_arr = pd.concat(
                [y_train, y_valid], axis=0).reset_index(drop=True).values
            for epoch in range(refit_epochs):
                t2 = time()
                self.unified_shuffle(x_index_combine_arr,
                                     x_value_combine_arr, y_combine_arr)
                total_batch = int(len(y_combine_arr) / self.batch_size)
                for i in range(total_batch):
                    x_index_batch, x_value_batch, y_batch = self.get_batch(
                        x_index_combine_arr, x_value_combine_arr,
                        y_train, self.batch_size, i)
                    self.fit_on_batch(x_index_batch, x_value_batch, y_batch)
                y_combine_pred = self.predict(
                    x_index_combine_arr, x_value_combine_arr)
                combine_result = self.eval_metric(
                    y_combine_arr, y_combine_pred)

                if refit_verbose > 0 and epoch % refit_verbose == 0:
                    print("[refit epoch = %d] combine-result=%.4f "
                          "training duration=[%.1f s]"
                          % (epoch + 1, combine_result, time() - t2))
                if abs(combine_result - best_train_score) < 0.001 \
                        or (self.bigger_is_better and
                            combine_result > best_train_score) \
                        or ((not self.bigger_is_better) and
                            combine_result < best_train_score):
                    break

        self._param = {weight_name: self.sess.run(weight_value)
                       for weight_name, weight_value in self.weights.items()}
        return self

    def predict(self, x_index_df, x_value_df):

        batch_index, dummy_y = 0, [1] * x_index_df.shape[0]
        x_index_batch, x_value_batch, _ = self.get_batch(
            x_index_df, x_value_df, dummy_y, self.batch_size, batch_index)
        y_prob = None

        while x_index_batch.shape[0] > 0:
            batch_len = x_index_batch.shape[0]
            feed_dict = {self.feat_index: x_index_batch,
                         self.feat_value: x_value_batch,
                         self.dropout_keep_deep: [1]*len(self.dropout_deep),
                         self.dropout_keep_fm: [1]*len(self.dropout_fm)}
            batch_output = self.sess.run(self.output, feed_dict=feed_dict)
            if batch_index == 0:
                y_prob = np.reshape(batch_output, (batch_len,))
            else:
                y_prob = np.concatenate((
                    y_prob, np.reshape(batch_output, (batch_len,))))
            batch_index += 1

            x_index_batch, x_value_batch, y_batch = self.get_batch(
                x_index_df, x_value_df, dummy_y, self.batch_size, batch_index)
        return y_prob

    def training_termination(self, valid_result, stop_threshold=5):

        if len(valid_result) > stop_threshold:
            if self.bigger_is_better:
                if valid_result[-1] < valid_result[-2] < \
                        valid_result[-3] < valid_result[-4] < valid_result[-5]:
                    return True
            else:
                if valid_result[-1] > valid_result[-2] > \
                        valid_result[-3] > valid_result[-4] > valid_result[-5]:
                    return True
        return False
