# Author: duguiming
# Description: TextRNN+attention网络结构
# Date: 2020-03-31
import os
import tensorflow as tf
from models.base_config import BaseConfig

base_dir = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))


class Config(BaseConfig):
    """TextRNN+Attention配置文件"""
    embedding_dim = 100
    seq_length = 600
    num_classes = 10
    vocab_size = 8000
    num_layers = 2
    hidden_dim = 128
    attention_size = 100
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
    tensorboard_dir = os.path.join(base_dir, 'result/tensorboard/textrnn_att/')
    best_model_dir = os.path.join(base_dir, 'result/best_model/textrnn_att/')


class TextRNNAtt(object):
    """TextRNN+attention模型"""
    def __init__(self, config):
        self.config = config
        self.input_x = tf.placeholder(tf.int32, [None, self.config.seq_length], name='input_x')
        self.input_y = tf.placeholder(tf.float32, [None, self.config.num_classes], name='input_y')
        self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')
        self.global_step = tf.Variable(0, trainable=False, name='global_step')
        self.l2_loss = tf.constant(0.0)
        self.rnn_att()

    def rnn_att(self):
        with tf.name_scope('Cell'):
            cell_fw = tf.contrib.rnn.BasicLSTMCell(self.config.hidden_dim)
            Cell_fw = tf.contrib.rnn.DropoutWrapper(cell_fw, self.keep_prob)

            cell_bw = tf.contrib.rnn.BasicLSTMCell(self.config.hidden_dim)
            Cell_bw = tf.contrib.rnn.DropoutWrapper(cell_bw, self.keep_prob)

        with tf.device('/cpu:0'):
            self.embedding = tf.get_variable('embedding', shape=[self.config.vocab_size, self.config.embedding_dim],
                                             initializer=tf.constant_initializer(self.config.pre_training))
            self.embedding_inputs = tf.nn.embedding_lookup(self.embedding, self.input_x)

        with tf.name_scope("BiRNN"):
            output, _ = tf.nn.bidirectional_dynamic_rnn(cell_fw=Cell_fw, cell_bw=Cell_bw, inputs=self.embedding_inputs,
                                                        dtype=tf.float32)
            output = tf.concat(output, 2)  # [batch_size, seq_length, 2*hidden_dim]

        with tf.name_scope("attention"):
            u_list = []
            seq_size = output.shape[1].value
            hidden_size = output.shape[2].value  # [2*hidden_dim]
            attention_w = tf.Variable(tf.truncated_normal([hidden_size, self.config.attention_size], stddev=0.1),
                                      name='attention_w')
            attention_u = tf.Variable(tf.truncated_normal([self.config.attention_size, 1], stddev=0.1), name='attention_u')
            attention_b = tf.Variable(tf.constant(0.1, shape=[self.config.attention_size]), name='attention_b')
            for t in range(seq_size):
                # u_t:[1,attention]
                u_t = tf.tanh(tf.matmul(output[:, t, :], attention_w) + tf.reshape(attention_b, [1, -1]))
                u = tf.matmul(u_t, attention_u)
                u_list.append(u)
            logit = tf.concat(u_list, axis=1)
            # u[seq_size:attention_z]
            weights = tf.nn.softmax(logit, name='attention_weights')
            # weight:[seq_size:1]
            out_final = tf.reduce_sum(output * tf.reshape(weights, [-1, seq_size, 1]), 1)
            # out_final:[batch,hidden_size]

        with tf.name_scope('dropout'):
            self.out_drop = tf.nn.dropout(out_final, keep_prob=self.keep_prob)

        with tf.name_scope('output'):
            w = tf.Variable(tf.truncated_normal([hidden_size, self.config.num_classes], stddev=0.1), name='w')
            b = tf.Variable(tf.zeros([self.config.num_classes]), name='b')
            self.logits = tf.matmul(self.out_drop, w) + b
            self.prob = tf.nn.softmax(self.logits)
            self.y_pred_cls = tf.argmax(tf.nn.softmax(self.logits), 1, name='predict')

        with tf.name_scope('loss'):
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.input_y)
            self.loss = tf.reduce_mean(cross_entropy)

        with tf.name_scope('optimizer'):
            # 退化学习率 learning_rate = lr*(0.9**(global_step/10);staircase=True表示每decay_steps更新梯度
            # learning_rate = tf.train.exponential_decay(self.config.lr, global_step=self.global_step,
            # decay_steps=10, decay_rate=self.config.lr_decay, staircase=True)
            # optimizer = tf.train.AdamOptimizer(learning_rate)
            # self.optimizer = optimizer.minimize(self.loss, global_step=self.global_step) #global_step 自动+1
            # no.2
            optimizer = tf.train.AdamOptimizer(self.config.learning_rate)
            gradients, variables = zip(*optimizer.compute_gradients(self.loss))  # 计算变量梯度，得到梯度值,变量
            gradients, _ = tf.clip_by_global_norm(gradients, self.config.clip)
            # 对g进行l2正则化计算，比较其与clip的值，如果l2后的值更大，让梯度*(clip/l2_g),得到新梯度
            self.optim = optimizer.apply_gradients(zip(gradients, variables), global_step=self.global_step)
            # global_step 自动+1

        with tf.name_scope('accuracy'):
            correct_pred = tf.equal(self.predict, tf.argmax(self.input_y, 1))
            self.acc = tf.reduce_mean(tf.cast(correct_pred, tf.float32), name='accuracy')
