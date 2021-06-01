"""
@author:duguiming
@description:训练词向量
"""
import logging
import time
import jieba
import sys
from gensim.models import word2vec

sys.path.append('..')

from DL.models.base_config import BaseConfig


class ReadData(object):
    def __init__(self, filenames):
        self.filenames = filenames

    def __iter__(self):
        for filename in self.filenames:
            with open(filename, 'r', encoding='utf-8') as f:
                for _, line in enumerate(f):
                    try:
                        line = line.strip()
                        _, line = line.split('\t', 1)
                        word = []
                        word.extend(jieba.cut(line))
                        yield word
                    except:
                        pass


def train_word2vec(filenames):
    """训练词向量模型"""
    t1 = time.time()
    texts = ReadData(filenames)
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    model = word2vec.Word2Vec(sentences=texts, size=100, window=5, min_count=1, workers=5)
    model.wv.save_word2vec_format(config.word2vec_save_path, binary=False)
    print("Training word2vec model cost %.3f seconds...\n" % (time.time() - t1))


if __name__ == "__main__":
    config = BaseConfig()
    filenames = [config.train_dir, config.test_dir, config.val_dir]
    train_word2vec(filenames)
