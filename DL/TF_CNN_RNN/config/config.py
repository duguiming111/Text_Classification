"""
@atthor:duguiming
@desceiption:配置文件
"""
import os
base_dir = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))


class PathConfig(object):
    """路径配置文件"""
    # 训练集、验证集和测试集地址
    train_dir = os.path.join(base_dir, 'data/cnews.train.txt')
    val_dir = os.path.join(base_dir, 'data/cnews.val.txt')
    test_dir = os.path.join(base_dir, 'data/cnews.test.txt')
    # 停用词表
    stopwords_dir = os.path.join(base_dir, 'data/stopwords.txt')
    # 字典路径
    vocab_dir = os.path.join(base_dir, 'data/vocab.txt')
    # 词向量路径
    word2vec_save_path = os.path.join(base_dir, 'result/word2vec.txt')
    vector_word_npz = os.path.join(base_dir, 'data/vector_word.npz')
    # tensorboard地址
    cnn_tensorboard_dir = os.path.join(base_dir, 'result/tensorboard/textcnn')
    rnn_tensorboard_dir = os.path.join(base_dir, 'result/tensorboard/textrnn')
    # 模型存放地址
    cnn_best_model_dir = os.path.join(base_dir, 'result/best_model/textcnn')
    rnn_best_model_dir = os.path.join(base_dir, 'result/best_model/textrnn')

class CNNConfig(object):
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


class RNNConfig(object):
    """RNN配置文件"""
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