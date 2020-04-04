# Author: duguiming
# Description: HAN网络结构
# Date: 2020-03-31
import os
import tensorflow as tf
from models.base_config import BaseConfig

base_dir = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))


class Config(BaseConfig):
    """Han配置文件"""
    name = "HAN"
    embedding_dim = 100
    seq_length = 600
    num_classes = 10
    vocab_size = 8000
    hidden_dim = 128
    num_sentences = 3
    rnn = "gru"
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
    tensorboard_dir = os.path.join(base_dir, 'result/tensorboard/han/')
    best_model_dir = os.path.join(base_dir, 'result/best_model/han/')


class HAN(object):
    """Han模型"""
    def __init__(self, config):
        self.config = config
        self.name = config.name
        self.input_x = tf.placeholder(tf.int32, [None, self.config.seq_length], name='input_x')
        self.input_y = tf.placeholder(tf.float32, [None, self.config.num_classes], name='input_y')
        self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')
        self.global_step = tf.Variable(0, trainable=False, name='global_step')
        self.l2_loss = tf.constant(0.0)
        self.han()

    def han(self):
        with tf.device('/cpu:0'):
            input_x = tf.split(self.input_x, self.config.num_sentences, axis=1)
            input_x = tf.stack(input_x, axis=1)
            self.embedding = tf.get_variable('embedding', shape=[self.config.vocab_size, self.config.embedding_dim],
                                             initializer=tf.constant_initializer(self.config.pre_training))
            self.embedding_inputs = tf.nn.embedding_lookup(self.embedding, input_x)
            self.sentence_len = int(self.config.seq_length / self.config.num_sentences)
            embedding_inputs_reshaped = tf.reshape(self.embedding_inputs,
                                                   shape=[-1, self.sentence_len, self.config.embedding_dim])

        # 词汇层
        with tf.name_scope("word_encoder"):
            (output_fw, output_bw) = self.bidirectional_rnn(embedding_inputs_reshaped, "word_encoder")
            # [batch_size*num_sentences,sentence_length,hidden_size * 2]
            word_hidden_state = tf.concat((output_fw, output_bw), 2)

        with tf.name_scope("word_attention"):
            # [batch_size*num_sentences, hidden_size * 2]
            sentence_vec = self.attention(word_hidden_state, "word_attention")

        # 句子层
        with tf.name_scope("sentence_encoder"):
            # [batch_size,num_sentences,hidden_size*2]
            sentence_vec = tf.reshape(sentence_vec, shape=[-1, self.config.num_sentences, self.config.hidden_dim * 2])
            output_fw, output_bw = self.bidirectional_rnn(sentence_vec, "sentence_encoder")
            # [batch_size*num_sentences,sentence_length,hidden_size * 2]
            sentence_hidden_state = tf.concat((output_fw, output_bw), 2)

        with tf.name_scope("sentence_attention"):
            # [batch_size, hidden_size * 2]
            doc_vec = self.attention(sentence_hidden_state, "sentence_attention")

        with tf.name_scope("dropout"):
            h_drop = tf.nn.dropout(doc_vec, self.keep_prob)

        with tf.name_scope('output'):
            # w = tf.Variable(tf.truncated_normal([hidden_size, self.config.num_classes], stddev=0.1), name='w')
            # b = tf.Variable(tf.zeros([self.config.num_classes]), name='b')
            # self.logits = tf.matmul(self.out_drop, w) + b
            self.logits = tf.layers.dense(h_drop, self.config.num_classes, name='fc2')
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
            correct_pred = tf.equal(self.y_pred_cls, tf.argmax(self.input_y, 1))
            self.acc = tf.reduce_mean(tf.cast(correct_pred, tf.float32), name='accuracy')

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

    def bidirectional_rnn(self, inputs, name):
        with tf.variable_scope(name):
            fw_cell = self.rnn_cell()
            fw_cell = tf.nn.rnn_cell.DropoutWrapper(fw_cell, output_keep_prob=self.config.dropout_keep_prob)
            bw_cell = self.rnn_cell()
            bw_cell = tf.nn.rnn_cell.DropoutWrapper(bw_cell, output_keep_prob=self.config.dropout_keep_prob)
            (output_fw, output_bw), states = tf.nn.bidirectional_dynamic_rnn(cell_fw=fw_cell,
                                                                             cell_bw=bw_cell,
                                                                             inputs=inputs,
                                                                             dtype=tf.float32)
        return output_fw, output_bw

    def attention(self, inputs, name):
        with tf.variable_scope(name):
            # 采用general形式计算权重
            hidden_vec = tf.layers.dense(inputs, self.config.hidden_dim * 2, activation=tf.nn.tanh, name='w_hidden')
            u_context = tf.Variable(tf.truncated_normal([self.config.hidden_dim * 2]), name='u_context')
            alpha = tf.nn.softmax(tf.reduce_sum(tf.multiply(hidden_vec, u_context),
                                                axis=2, keep_dims=True), dim=1)

            # 对隐藏状态进行加权
            attention_output = tf.reduce_sum(tf.multiply(inputs, alpha), axis=1)

        return attention_output