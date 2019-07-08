# Author:duguiming
# Description:数据处理
# Date:2019-07-08
import jieba
import numpy as np
import tensorflow.contrib.keras as kr
from collections import Counter
from config.config import PathConfig

config = PathConfig()


class DataProcess(object):
    def __init__(self):
        pass

    def read_data(self, filename):
        """读取数据"""
        # 读取停用词
        stopwords = list()
        with open(config.stopwords_dir, 'r', encoding='utf-8') as f:
            for word in f.readlines():
                stopwords.append(word[:-1])
        labels = list()
        texts = list()
        with open(filename, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            for line in lines:
                label, text = line.split('\t', 1)
                labels.append(label)
                # 去掉空格和回车
                text = text.replace('\n', '')
                text = text.replace(' ', '')
                texts.append([word for word in jieba.cut(text) if word not in stopwords])
        return labels, texts

    def build_vocab(self, filenames, vocab_dir, vocab_size=8000):
        """构建词典"""
        all_data = []
        for filename in filenames:
            _, data = self.read_data(filename)
            for text in data:
                all_data.extend(text)
        counter = Counter(all_data)
        count_pairs = counter.most_common(vocab_size-1)
        words, _ = list(zip(*count_pairs))
        words = ['<PAD>'] + list(words)
        with open(vocab_dir, 'w', encoding='utf-8') as f:
            f.write('\n'.join(words))

    def read_vocab(self, vocab_dir):
        """读取词典"""
        with open(vocab_dir, 'r', encoding='utf-8') as f:
            words = f.read().strip().split('\n')
        word2id = dict(zip(words, range(len(words))))
        return words, word2id

    def read_category(self):
        """读取类别"""
        categories = ['体育', '财经', '房产', '家居', '教育', '科技', '时尚', '时政', '游戏', '娱乐']
        cat2id = dict(zip(categories, range(len(categories))))
        return categories, cat2id

    def process_file(self, filename, word_to_id, cat_to_id, max_length=600):
        """将文件转换为id表示"""
        labels, texts = self.read_data(filename)
        data_id, label_id = [], []
        for i in range(len(texts)):
            data_id.append([word_to_id[x] for x in texts[i] if x in word_to_id])
            label_id.append(cat_to_id[labels[i]])
        # 使用keras提供的pad_sequences来将文本pad为固定长度
        x_pad = kr.preprocessing.sequence.pad_sequences(data_id, max_length)
        y_pad = kr.utils.to_categorical(label_id, num_classes=len(cat_to_id))  # 将标签转换为one-hot表示
        return x_pad, y_pad

    def batch_iter(self, x, y, batch_size=64):
        """生成迭代器"""
        data_len = len(x)
        num_batch = int((data_len - 1) / batch_size) + 1
        indices = np.random.permutation(np.arange(data_len))
        x_shuffle = x[indices]
        y_shuffle = y[indices]
        for i in range(num_batch):
            start_id = i * batch_size
            end_id = min((i + 1) * batch_size, data_len)
            yield x_shuffle[start_id:end_id], y_shuffle[start_id:end_id]

    def export_word2vec_vectors(self, vocab, word2vec_dir, trimmed_filename):
        """将字典转化为向量模式"""
        file_r = open(word2vec_dir, 'r', encoding='utf-8')
        line = file_r.readline()
        voc_size, vec_dim = map(int, line.split(' '))
        embeddings = np.zeros([len(vocab), vec_dim])
        line = file_r.readline()
        while line:
            try:
                items = line.split(' ')
                word = items[0]
                vec = np.asarray(items[1:], dtype='float32')
                if word in vocab:
                    word_idx = vocab[word]
                    embeddings[word_idx] = np.asarray(vec)
            except:
                pass
            line = file_r.readline()
        np.savez_compressed(trimmed_filename, embeddings=embeddings)

    def get_training_word2vec_vectors(self, filename):
        """读取词向量矩阵"""
        with np.load(filename) as data:
            return data["embeddings"]

