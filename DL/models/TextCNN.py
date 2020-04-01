# Author: duguiming
# Description: TextCNN网络结构
# Date: 2020-03-31
import os
import tensorflow as tf
from models.base_config import BaseConfig

base_dir = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))


class Config(BaseConfig):
    """CNN配置文件"""
    embedding_dim = 100
    seq_length = 600
    num_classes = 10
    num_filters = 256
    filter_size = [2, 3, 4]
    hidden_dim = 128
    dropout_keep_prob = 0.5
    learning_rate = 1e-3
    clip = 6.0
    batch_size = 64
    num_epochs = 10
    print_per_batch = 100
    save_per_batch = 10
    vocab_size = 8000
    pre_training = None
    l2_reg_lambda = 0.01
    # 路径
    tensorboard_dir = os.path.join(base_dir, 'result/tensorboard/textcnn/')
    best_model_dir = os.path.join(base_dir, 'result/best_model/textcnn/')


class TextCNN(object):
    """CNN模型"""
    def __init__(self, config):
        self.config = config
        self.input_x = tf.placeholder(tf.int32, shape=[None, self.config.seq_length], name='input_x')
        self.input_y = tf.placeholder(tf.int32, shape=[None, self.config.num_classes], name='input_y')
        self.keep_prob = tf.placeholder(tf.float32, name='dropout')
        self.global_step = tf.Variable(0, trainable=False, name='global_step')
        self.l2_loss = tf.constant(0.0)
        self.cnn()

    def cnn(self):
        with tf.device('/cpu:0'):
            self.embedding = tf.get_variable('embedding', shape=[self.config.vocab_size, self.config.embedding_dim],
                                             initializer=tf.constant_initializer(self.config.pre_training))
            self.embedding_inputs = tf.nn.embedding_lookup(self.embedding, self.input_x)
            self.embedding_inputs_expanded = tf.expand_dims(self.embedding_inputs, -1)

        with tf.name_scope('cnn'):
            pooled_outputs = list()
            for i, filter_size in enumerate(self.config.filter_size):
                with tf.name_scope("conv-maxpool-%s" % filter_size):
                    kernel_shape = [filter_size, self.config.embedding_dim, 1, self.config.num_filters]
                    W = tf.Variable(tf.truncated_normal(kernel_shape, stddev=0.1), name='W')
                    b = tf.Variable(tf.constant(0.1, shape=[self.config.num_filters]), name='b')
                    conv = tf.nn.conv2d(
                        self.embedding_inputs_expanded,
                        W,
                        strides=[1, 1, 1, 1],
                        padding='VALID',
                        name='conv'
                    )
                    h = tf.nn.relu(tf.nn.bias_add(conv, b), name='relu')
                    pooled = tf.nn.max_pool(
                        h,
                        ksize=[1, self.config.seq_length-filter_size+1, 1, 1],
                        strides=[1, 1, 1, 1],
                        padding='VALID',
                        name='pool'
                    )
                    pooled_outputs.append(pooled)
            num_filters_total = self.config.num_filters*len(self.config.filter_size)
            self.h_pool = tf.concat(pooled_outputs, 3)
            self.outputs = tf.reshape(self.h_pool, [-1, num_filters_total])

        with tf.name_scope('dropout'):
            self.final_output = tf.nn.dropout(self.outputs, self.keep_prob)

        with tf.name_scope('output'):
            fc_w = tf.get_variable('fc_w', shape=[self.final_output.shape[1], self.config.num_classes],
                                   initializer=tf.contrib.layers.xavier_initializer())
            fc_b = tf.Variable(tf.constant(0.1, shape=[self.config.num_classes]), name='fc_b')
            self.logits = tf.matmul(self.final_output, fc_w) + fc_b
            self.prob = tf.nn.softmax(self.logits)
            self.y_pred_cls = tf.argmax(self.logits, 1, name='prediction')

        with tf.name_scope('loss'):
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.input_y)
            self.l2_loss += tf.nn.l2_loss(fc_w)
            self.l2_loss += tf.nn.l2_loss(fc_b)
            self.loss = tf.reduce_mean(cross_entropy) + self.config.l2_reg_lambda*self.l2_loss
            # self.loss = tf.reduce_mean(cross_entropy)

        with tf.name_scope('optimizer'):
            optimizer = tf.train.AdamOptimizer(self.config.learning_rate)
            gradients, variables = zip(*optimizer.compute_gradients(self.loss))
            gradients, _ = tf.clip_by_global_norm(gradients, self.config.clip)
            self.optim = optimizer.apply_gradients(zip(gradients, variables), global_step=self.global_step)

        with tf.name_scope('accuracy'):
            correct_pred = tf.equal(tf.argmax(self.input_y, 1), self.y_pred_cls)
            self.acc = tf.reduce_mean(tf.cast(correct_pred, tf.float32))