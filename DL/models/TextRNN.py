# Author: duguiming
# Description: TextRNN网络结构
# Date: 2020-03-31
import os
import tensorflow as tf
from DL.models.base_config import BaseConfig

base_dir = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))


class Config(BaseConfig):
    """RNN配置文件"""
    name = "TextRNN"
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
    tensorboard_dir = os.path.join(base_dir, 'result/tensorboard/textrnn/')
    best_model_dir = os.path.join(base_dir, 'result/best_model/textrnn/')


class TextRNN(object):
    """RNN模型"""
    def __init__(self, config):
        self.config = config
        self.name = config.name
        self.input_x = tf.placeholder(tf.int32, [None, self.config.seq_length], name='input_x')
        self.input_y = tf.placeholder(tf.float32, [None, self.config.num_classes], name='input_y')
        self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')
        self.global_step = tf.Variable(0, trainable=False, name='global_step')
        self.l2_loss = tf.constant(0.0)
        self.rnn()

    def rnn(self):
        # 选择rnn的结构
        def lstm_cell():
            return tf.contrib.rnn.BasicLSTMCell(self.config.hidden_dim, state_is_tuple=True)

        def gru_cell():
            return tf.contrib.rnn.GRUCell(self.config.hidden_dim)

        def dropout():
            if self.config.rnn == 'lstm':
                cell = lstm_cell()
            else:
                cell = gru_cell()
            return tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=self.keep_prob)

        with tf.device('/cpu:0'):
            self.embedding = tf.get_variable('embedding', shape=[self.config.vocab_size, self.config.embedding_dim],
                                             initializer=tf.constant_initializer(self.config.pre_training))
            self.embedding_inputs = tf.nn.embedding_lookup(self.embedding, self.input_x)
            # self.embedding_inputs_expanded = tf.expand_dims(self.embedding_inputs, -1)

        with tf.name_scope('rnn'):
            cells = [dropout() for _ in range(self.config.num_layers)]
            rnn_cell = tf.contrib.rnn.MultiRNNCell(cells, state_is_tuple=True)
            _outputs, _ = tf.nn.dynamic_rnn(cell=rnn_cell, inputs=self.embedding_inputs, dtype=tf.float32)
            last = _outputs[:, -1, :]

        with tf.name_scope('dropout'):
            fc = tf.layers.dense(last, self.config.hidden_dim, name='fc1')
            fc = tf.contrib.layers.dropout(fc, self.keep_prob)
            fc = tf.nn.relu(fc)

        with tf.name_scope('output'):
            # self.logits = tf.layers.dense(fc, self.config.num_classes, name='fc2')
            # self.y_pred_cls = tf.argmax(tf.nn.softmax(self.logits), 1)
            fc_w = tf.get_variable('fc_w', shape=[fc.shape[1], self.config.num_classes],
                                   initializer=tf.contrib.layers.xavier_initializer())
            fc_b = tf.Variable(tf.constant(0.1, shape=[self.config.num_classes]), name='fc_b')
            self.logits = tf.matmul(fc, fc_w) + fc_b
            self.prob = tf.nn.softmax(self.logits)
            self.y_pred_cls = tf.argmax(self.logits, 1, name='prediction')

        with tf.name_scope('loss'):
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.input_y)
            self.l2_loss += tf.nn.l2_loss(fc_w)
            self.l2_loss += tf.nn.l2_loss(fc_b)
            self.loss = tf.reduce_mean(cross_entropy) + self.config.l2_reg_lambda * self.l2_loss

        with tf.name_scope('optimizer'):
            optimizer = tf.train.AdamOptimizer(self.config.learning_rate)
            gradients, variables = zip(*optimizer.compute_gradients(self.loss))
            gradients, _ = tf.clip_by_global_norm(gradients, self.config.clip)
            self.optim = optimizer.apply_gradients(zip(gradients, variables), global_step=self.global_step)

        with tf.name_scope('accuracy'):
            correct_pred = tf.equal(tf.argmax(self.input_y, 1), self.y_pred_cls)
            self.acc = tf.reduce_mean(tf.cast(correct_pred, tf.float32))