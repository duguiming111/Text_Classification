# Author: duguiming
# Description: TextRCNN网络结构
# Date: 2020-03-31
import os
import tensorflow as tf
from models.base_config import BaseConfig

base_dir = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))


class Config(BaseConfig):
    """TextRCNN配置文件"""
    name = "TextRCNN"
    embedding_dim = 100
    seq_length = 600
    num_classes = 10
    vocab_size = 8000
    num_layers = 2
    hidden_dim = 128
    rnn = 'gru'
    dropout_keep_prob = 0.8
    learning_rate = 1e-3
    batch_size = 64
    num_epochs = 10
    print_per_batch = 100
    save_per_batch = 10
    pre_training = None
    clip = 6.0
    l2_reg_lambda = 0.01

    # 路径
    tensorboard_dir = os.path.join(base_dir, 'result/tensorboard/rcnn/')
    best_model_dir = os.path.join(base_dir, 'result/best_model/rcnn/')


class TextRCNN(object):
    """TextRCNN模型"""
    def __init__(self, config):
        self.config = config
        self.name = config.name
        self.input_x = tf.placeholder(tf.int32, [None, self.config.seq_length], name='input_x')
        self.input_y = tf.placeholder(tf.float32, [None, self.config.num_classes], name='input_y')
        self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')
        self.global_step = tf.Variable(0, trainable=False, name='global_step')
        self.l2_loss = tf.constant(0.0)
        self.rcnn()

    def rcnn(self):
        with tf.device('/cpu:0'):
            self.embedding = tf.get_variable('embedding', shape=[self.config.vocab_size, self.config.embedding_dim],
                                             initializer=tf.constant_initializer(self.config.pre_training))
            self.embedding_inputs = tf.nn.embedding_lookup(self.embedding, self.input_x)

        with tf.name_scope("bi-rnn"):
            fw_cell = self.rnn_cell()
            fw_cell = tf.nn.rnn_cell.DropoutWrapper(fw_cell, output_keep_prob=self.keep_prob)
            bw_cell = self.rnn_cell()
            bw_cell = tf.nn.rnn_cell.DropoutWrapper(bw_cell, output_keep_prob=self.keep_prob)
            (self.output_fw, self.output_bw), states = tf.nn.bidirectional_dynamic_rnn(cell_fw=fw_cell,
                                                                                       cell_bw=bw_cell,
                                                                                       inputs=self.embedding_inputs,
                                                                                       dtype=tf.float32)

        with tf.name_scope("context"):
            shape = [tf.shape(self.output_fw)[0], 1, tf.shape(self.output_fw)[2]]
            self.c_left = tf.concat([tf.zeros(shape), self.output_fw[:, :-1]], axis=1, name="context_left")
            self.c_right = tf.concat([self.output_bw[:, 1:], tf.zeros(shape)], axis=1, name="context_right")

        with tf.name_scope("word-representation"):
            self.x = tf.concat([self.c_left, self.embedding_inputs, self.c_right], axis=2, name="x")
            # embedding_size = 2 * self.config.context_embedding_size + self.config.word_embedding_size
            embedding_size = int(self.x.get_shape()[2])

        with tf.name_scope("text-representation"):
            W2 = tf.Variable(tf.random_uniform([embedding_size, self.config.hidden_dim], -1.0, 1.0), name="W2")
            b2 = tf.Variable(tf.constant(0.1, shape=[self.config.hidden_dim]), name="b2")
            self.y2 = tf.tanh(tf.einsum('aij,jk->aik', self.x, W2) + b2)

        with tf.name_scope("max-pooling"):
            self.y3 = tf.reduce_max(self.y2, axis=1)

        with tf.name_scope("output"):
            W4 = tf.get_variable("W4", shape=[self.config.hidden_dim, self.config.num_classes],
                                 initializer=tf.contrib.layers.xavier_initializer())
            b4 = tf.Variable(tf.constant(0.1, shape=[self.config.num_classes]), name="b4")
            self.l2_loss += tf.nn.l2_loss(W4)
            self.l2_loss += tf.nn.l2_loss(b4)
            self.logits = tf.matmul(self.y3, W4) + b4
            self.prob = tf.nn.softmax(self.logits)
            self.y_pred_cls = tf.argmax(self.logits, 1, name="predictions")

        # Calculate mean cross-entropy loss
        with tf.name_scope("loss"):
            losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.input_y)
            self.loss = tf.reduce_mean(losses) + self.config.l2_reg_lambda * self.l2_loss

        with tf.name_scope('optimizer'):
            optimizer = tf.train.AdamOptimizer(self.config.learning_rate)
            gradients, variables = zip(*optimizer.compute_gradients(self.loss))
            gradients, _ = tf.clip_by_global_norm(gradients, self.config.clip)
            self.optim = optimizer.apply_gradients(zip(gradients, variables), global_step=self.global_step)

        # Accuracy
        with tf.name_scope("accuracy"):
            correct_pred = tf.equal(self.y_pred_cls, tf.argmax(self.input_y, axis=1))
            self.acc = tf.reduce_mean(tf.cast(correct_pred, tf.float32), name="accuracy")

    def rnn_cell(self):
        """获取rnn的cell，可选RNN、LSTM、GRU"""
        if self.config.rnn == "vanilla":
            return tf.nn.rnn_cell.BasicRNNCell(self.config.hidden_dim)
        elif self.config.rnn == "lstm":
            return tf.nn.rnn_cell.BasicLSTMCell(self.config.hidden_dim)
        elif self.config.rnn == "gru":
            return tf.nn.rnn_cell.GRUCell(self.config.hidden_dim)
        else:
            raise Exception("rnn_type must be vanilla、lstm or gru!")