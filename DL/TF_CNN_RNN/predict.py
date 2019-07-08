"""
@dauthor:duguiming
@description:用训练好的模型进行预测
"""
import heapq
import tensorflow as tf

from model.text_model import TextCNN
from config.config import PathConfig, CNNConfig
from data_helper.data_process import DataProcess


class CNNModel(object):
    def __init__(self):
        self.config = CNNConfig()
        self.categories, self.cat2id = dp.read_category()
        self.config.pre_training = dp.get_training_word2vec_vectors(pathconfig.vector_word_npz)
        self.model = TextCNN(self.config)
        _, self.word2id = dp.read_vocab(pathconfig.vocab_dir)

    def predict(self, sentence):
        input_x = dp.process_file(sentence, self.word2id, max_length=self.config.seq_length)
        feed_dict = {
            self.model.input_x: input_x,
            self.model.keep_prob: 1
        }
        session = tf.Session()
        session.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        saver.restore(sess=session, save_path=pathconfig.cnn_best_model_dir)
        y_pred_cls = session.run(self.model.prob, feed_dict=feed_dict)
        return self.categories[y_pred_cls[0]]


if __name__ == '__main__':
    pathconfig = PathConfig()
    dp = DataProcess()
    print('predict random five samples in test data.... ')
    import random
    sentences = list()
    labels = list()
    with open(pathconfig.test_dir, 'r', encoding='utf-8') as f:
        sample = random.sample(f.readlines(), 5)
        for line in sample:
            try:
                line = line.rstrip().split('\t')
                assert len(line) == 2
                sentences.append(line[1])
                labels.append(line[0])
            except:
                pass
    cnn_model = CNNModel()
    cat = list()
    for sentence in sentences:
        cat.append(cnn_model.predict(sentence))
    for i, sentence in enumerate(sentences, 0):
        print('----------------------the text-------------------------')
        print(sentence[:50]+'....')
        print('the orginal label:{}'.format(labels[i]))
        print('the predict label:{}'.format(cat[i]))

