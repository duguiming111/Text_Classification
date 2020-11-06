# Author: duguiming
# Description: Transformer网络结构
# Date: 2020-03-31
import os
import numpy as np
import tensorflow as tf
from DL.models.base_config import BaseConfig

base_dir = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))


class Config(BaseConfig):
    """Transformer配置文件"""
    name = "Transformer"
    embedding_dim = 100
    seq_length = 120
    num_classes = 10
    vocab_size = 8000
    num_units = 100  # query,key,value的维度
    ffn_dim = 2048
    num_heads = 4
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
    tensorboard_dir = os.path.join(base_dir, 'result/tensorboard/transformer/')
    best_model_dir = os.path.join(base_dir, 'result/best_model/transformer/')


class Transformer(object):
    """Transformer模型"""
    def __init__(self, config):
        self.config = config
        self.name = config.name
        self.input_x = tf.placeholder(tf.int32, [None, self.config.seq_length], name='input_x')
        self.input_y = tf.placeholder(tf.float32, [None, self.config.num_classes], name='input_y')
        self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')
        self.global_step = tf.Variable(0, trainable=False, name='global_step')
        self.l2_loss = tf.constant(0.0)
        self.transformer()

    def transformer(self):
        with tf.device('/cpu:0'):
            self.embedding = tf.get_variable('embedding', shape=[self.config.vocab_size, self.config.embedding_dim],
                                             initializer=tf.constant_initializer(self.config.pre_training))
            self.embedding_inputs = tf.nn.embedding_lookup(self.embedding, self.input_x)

        #  Positional Encoding
        with tf.name_scope("positional_encoding"):
            positional_output = self.positional_encoding(self.embedding_inputs)

        with tf.name_scope('dropout'):
            positional_output = tf.nn.dropout(positional_output, self.keep_prob)

        # 注意力机制
        with tf.name_scope('attention'):
            attention_output = self.multihead_attention(positional_output)

        # Residual connection
        attention_output += positional_output

        # [batch_size, sequence_length, num_units]
        outputs = self.layer_normalize(attention_output)  # LN

        # feedforward
        feedforward_outputs = self.feedforward(outputs)

        # Residual connection
        feedforward_outputs += outputs

        # LN
        feedforward_outputs = self.layer_normalize(feedforward_outputs)
        self.outputs = tf.reduce_mean(feedforward_outputs, axis=1)

        with tf.name_scope('output'):
            w = tf.Variable(tf.truncated_normal([self.config.num_units, self.config.num_classes], stddev=0.1), name='w')
            b = tf.Variable(tf.zeros([self.config.num_classes]), name='b')
            self.logits = tf.matmul(self.outputs, w) + b
            self.prob = tf.nn.softmax(self.logits)
            self.y_pred_cls = tf.argmax(tf.nn.softmax(self.logits), 1, name='predict')

        with tf.name_scope('loss'):
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.input_y)
            self.loss = tf.reduce_mean(cross_entropy)

        with tf.name_scope('optimizer'):
            optimizer = tf.train.AdamOptimizer(self.config.learning_rate)
            gradients, variables = zip(*optimizer.compute_gradients(self.loss))  # 计算变量梯度，得到梯度值,变量
            gradients, _ = tf.clip_by_global_norm(gradients, self.config.clip)
            # 对g进行l2正则化计算，比较其与clip的值，如果l2后的值更大，让梯度*(clip/l2_g),得到新梯度
            self.optim = optimizer.apply_gradients(zip(gradients, variables), global_step=self.global_step)
            # global_step 自动+1

        with tf.name_scope('accuracy'):
            correct_pred = tf.equal(self.y_pred_cls, tf.argmax(self.input_y, 1))
            self.acc = tf.reduce_mean(tf.cast(correct_pred, tf.float32), name='accuracy')

    def positional_encoding(self, embedding_inputs):
        """
        增加位置信息
        """
        positional_ind = tf.tile(tf.expand_dims(tf.range(self.config.seq_length), 0),
                                 [self.config.batch_size, 1])  # [batch_size, sequence_length]
        # [sequence_length,embedding_size]
        position_enc = np.array(
            [[pos / np.power(10000, 2. * i / self.config.embedding_dim) for i in range(self.config.embedding_dim)]
             for pos in range(self.config.seq_length)])
        position_enc[:, 0::2] = np.sin(position_enc[:, 0::2])
        position_enc[:, 1::2] = np.cos(position_enc[:, 1::2])
        lookup_table = tf.convert_to_tensor(position_enc, dtype=tf.float32)
        # [batch_size,sequence_length,embedding_size]
        positional_output = tf.nn.embedding_lookup(lookup_table, positional_ind)
        positional_output += embedding_inputs
        return positional_output

    def multihead_attention(self, attention_inputs):
        """
        注意力机制
        """
        # [batch_size,sequence_length, num_units]
        Q = tf.keras.layers.Dense(self.config.num_units)(attention_inputs)
        K = tf.keras.layers.Dense(self.config.num_units)(attention_inputs)
        V = tf.keras.layers.Dense(self.config.num_units)(attention_inputs)

        # 将Q/K/V分成多头
        # Q_/K_/V_.shape = [batch_size*num_heads,sequence_length,num_units/num_heads]
        Q_ = tf.concat(tf.split(Q, self.config.num_heads, axis=2), axis=0)
        K_ = tf.concat(tf.split(K, self.config.num_heads, axis=2), axis=0)
        V_ = tf.concat(tf.split(V, self.config.num_heads, axis=2), axis=0)

        # 计算Q与K的相似度
        # tf.transpose(K_,[0,2,1])是对矩阵K_转置
        # similarity.shape = [batch_size*num_heads,sequence_length,sequence_length]
        similarity = tf.matmul(Q_, tf.transpose(K_, [0, 2, 1]))
        similarity = similarity / (K_.get_shape().as_list()[-1] ** 0.5)

        pad_mask = self.padding_mask(self.input_x)
        pad_mask = tf.tile(pad_mask, [self.config.num_heads, 1, 1])
        paddings = tf.ones_like(similarity) * (-2 ** 32 + 1)
        similarity = tf.where(tf.equal(pad_mask, False), paddings, similarity)
        similarity = tf.nn.softmax(similarity)
        similarity = tf.nn.dropout(similarity, self.keep_prob)
        # [batch_size*num_heads,sequence_length,sequence_length]
        outputs = tf.matmul(similarity, V_)
        outputs = tf.concat(tf.split(outputs, self.config.num_heads, axis=0), axis=2)
        return outputs

    def padding_mask(self, inputs):
        pad_mask = tf.equal(inputs, 0)
        # [batch_size,sequence_length,sequence_length]
        pad_mask = tf.tile(tf.expand_dims(pad_mask, axis=1), [1, self.config.seq_length, 1])
        return pad_mask

    def feedforward(self, inputs):
        params = {"inputs": inputs, "filters": self.config.ffn_dim, "kernel_size": 1, "activation": tf.nn.relu,
                  "use_bias": True}
        # 相当于 [batch_size*sequence_length,num_units]*[num_units,ffn_dim]，在reshape成[batch_size,sequence_length,num_units]
        # [batch_size,sequence_length,ffn_dim]
        outputs = tf.layers.conv1d(**params)
        params = {"inputs": outputs, "filters": self.config.num_units, "kernel_size": 1, "activation": None, "use_bias": True}
        # [batch_size,sequence_length,num_units]
        outputs = tf.layers.conv1d(**params)
        return outputs

    def layer_normalize(self, inputs, epsilon=1e-8):
        # [batch_size,sequence_length,num_units]
        inputs_shape = inputs.get_shape()
        params_shape = inputs_shape[-1:]  # num_units
        # 沿轴-1求均值和方差(也就是沿轴num_units)
        # mean/variance.shape = [batch_size,sequence_length]
        mean, variance = tf.nn.moments(inputs, [-1], keep_dims=True)  # LN
        # mean, variance = tf.nn.moments(inputs,[-2,-1],keep_dims=True) # BN
        beta = tf.Variable(tf.zeros(params_shape))
        gamma = tf.Variable(tf.ones(params_shape))
        normalized = (inputs - mean) / ((variance + epsilon) ** (.5))
        # [batch_size,sequence_length,num_units]
        outputs = gamma * normalized + beta
        return outputs