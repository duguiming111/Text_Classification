# Author: duguiming
# Description: DPCNN网络结构
# Date: 2020-03-31
import os
import tensorflow as tf
from models.base_config import BaseConfig

base_dir = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))


class Config(BaseConfig):
    embedding_dim = 100
    seq_length = 600
    num_classes = 10
    vocab_size = 8000
    num_filters = 250
    kernel_size = 3
    hidden_dim = 128
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
    tensorboard_dir = os.path.join(base_dir, 'result/tensorboard/dpcnn/')
    best_model_dir = os.path.join(base_dir, 'result/best_model/dpcnn/')


class DPCNN(object):
    """DPCNN模型"""

    def __init__(self, config):
        self.config = config
        self.input_x = tf.placeholder(tf.int32, [None, self.config.seq_length], name='input_x')
        self.input_y = tf.placeholder(tf.int32, [None, self.config.num_classes], name='input_y')
        self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')
        self.global_step = tf.Variable(0, trainable=False, name='global_step')
        self.l2_loss = tf.constant(0.0)
        self.dpcnn()

    def dpcnn(self):
        with tf.device('/cpu:0'):
            self.embedding = tf.get_variable('embedding', shape=[self.config.vocab_size, self.config.embedding_dim],
                                             initializer=tf.constant_initializer(self.config.pre_training))
            self.embedding_inputs = tf.nn.embedding_lookup(self.embedding, self.input_x)
            self.embedding_inputs_expanded = tf.expand_dims(self.embedding_inputs, -1)

        with tf.name_scope("pre_activation"):
            self.region_embedding = tf.layers.conv2d(self.embedding_inputs_expanded, self.config.num_filters,
                                                     [self.config.kernel_size, self.config.embedding_dim])
            self.pre_activation = tf.nn.relu(self.region_embedding, name='preactivation')

        with tf.name_scope("han"):
            # first layer
            conv0_1 = tf.layers.conv2d(self.pre_activation, self.config.num_filters, self.config.kernel_size,
                                       padding='same', activation=tf.nn.relu)
            conv0_2 = tf.layers.conv2d(conv0_1, self.config.num_filters, self.config.kernel_size,
                                       padding='same', activation=tf.nn.relu)

            # second layer
            conv1_1 = conv0_2 + self.pre_activation

            pool = tf.pad(conv1_1, paddings=[[0, 0], [0, 1], [0, 0], [0, 0]])
            pool = tf.nn.max_pool(pool, [1, 3, 1, 1], strides=[1, 2, 1, 1], padding='VALID')

            conv1_2 = tf.layers.conv2d(pool, self.config.num_filters, self.config.kernel_size,
                                       padding='same', activation=tf.nn.relu)
            conv1_3 = tf.layers.conv2d(conv1_2, self.config.num_filters, self.config.kernel_size,
                                       padding='same', activation=tf.nn.relu)

            # resdul
            conv2_1 = conv1_3 + pool
            pool_size = int((self.config.seq_length - 3 + 1) / 2)
            conv2_2 = tf.layers.max_pooling1d(tf.squeeze(conv2_1, [2]), pool_size, 1)
            conv2_3 = tf.squeeze(conv2_2, [1])  # [batch,250]

        with tf.name_scope('dropout'):
            self.output = tf.nn.dropout(conv2_3, self.keep_prob)

        with tf.name_scope('output'):
            fc_w = tf.get_variable('fc_w', shape=[self.output.shape[1], self.config.num_classes],
                                   initializer=tf.contrib.layers.xavier_initializer())
            fc_b = tf.Variable(tf.constant(0.1, shape=[self.config.num_classes]), name='fc_b')
            self.logits = tf.matmul(self.output, fc_w) + fc_b
            self.prob = tf.nn.softmax(self.logits)
            self.y_pred_cls = tf.argmax(tf.nn.softmax(self.logits), 1, name="pred")

        with tf.name_scope('loss'):
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.input_y)
            self.l2_loss += tf.nn.l2_loss(fc_w)
            self.l2_loss += tf.nn.l2_loss(fc_b)
            self.loss = tf.reduce_mean(cross_entropy, name="loss")

        with tf.name_scope('optimizer'):
            optimizer = tf.train.AdamOptimizer(self.config.learning_rate)
            gradients, variables = zip(*optimizer.compute_gradients(self.loss))
            gradients, _ = tf.clip_by_global_norm(gradients, self.config.clip)
            self.optim = optimizer.apply_gradients(zip(gradients, variables), global_step=self.global_step)

        with tf.name_scope('accuracy'):
            correct_pred = tf.equal(tf.argmax(self.input_y, 1), self.y_pred_cls)
            self.acc = tf.reduce_mean(tf.cast(correct_pred, tf.float32), name="acc")