# AUthor: duguiming
# Description: 基础的配置项目
# Date: 2020-03-31
import os

base_dir = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))


class BaseConfig(object):
    # 训练集、验证集和测试集地址
    train_dir = os.path.join(base_dir, 'data/cnews/cnews.train.txt')
    val_dir = os.path.join(base_dir, 'data/cnews/cnews.val.txt')
    test_dir = os.path.join(base_dir, 'data/cnews/cnews.test.txt')
    # 停用词表
    stopwords_dir = os.path.join(base_dir, 'data/cnews/stopwords.txt')
    # 字典路径
    vocab_dir = os.path.join(base_dir, 'data/cnews/vocab.txt')
    # 词向量路径
    word2vec_save_path = os.path.join(base_dir, 'result/word2vec.txt')
    vector_word_npz = os.path.join(base_dir, 'data/cnews/vector_word.npz')
