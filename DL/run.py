# Author: duguiming
# Description: 运行
# Date: 2020-3-31
import os
import argparse
from importlib import import_module
from data_helper.data_process import DataProcess
from train_val_test import train, test


parser = argparse.ArgumentParser(description='Chinese Text Classification')
parser.add_argument('--model', type=str, required=True, help='choose a model: TextCNN, TextRNN, TextRCNN, TextRNN_Att, '
                                                             'DPCNN, Transformer')
parser.add_argument('--mode', type=str, required=True, help='train or test')
parser.add_argument('--embedding', default='pre_trained', type=str, help='random or pre_trained')
parser.add_argument('--word', default=True, type=bool, help='True for word, False for char')
args = parser.parse_args()


if __name__ == '__main__':
    mode = args.mode
    model_name = args.model
    pkg = import_module('models.' + model_name)
    config = pkg.Config()

    dp = DataProcess()
    # 构建词表
    filenames = [config.train_dir, config.val_dir, config.test_dir]
    if not os.path.exists(config.vocab_dir):
        dp.build_vocab(filenames, config.vocab_dir, config.vocab_size)
    # 读取词表和类别
    categories, cat2id = dp.read_category()
    words, word2id = dp.read_vocab(config.vocab_dir)
    # 转化为向量
    if not os.path.exists(config.vector_word_npz):
        dp.export_word2vec_vectors(word2id, config.word2vec_save_path, config.vector_word_npz)
    config.pre_training = dp.get_training_word2vec_vectors(config.vector_word_npz)

    # 构造模型
    if model_name == "TextCNN":
        model = pkg.TextCNN(config)
    elif model_name == "TextRNN":
        model = pkg.TextRNN(config)
    elif model_name == "TextRCNN":
        model = pkg.TextRCNN(config)
    elif model_name == "HAN":
        model = pkg.HAN(config)
    elif model_name == "DPCNN":
        model = pkg.DPCNN(config)
    elif model_name == "TextRNN_Att":
        model = pkg.TextRNNAtt(config)
    elif model_name == "Transformer":
        model = pkg.Transformer(config)
    else:
        print("model选项错误，默认TextCNN")
        model = pkg.TextCNN(config)
    # 训练/测试
    if mode == "train":
        train(model, config, word2id, cat2id)
    elif mode == "test":
        test(model, config, word2id, cat2id, categories)
    else:
        print("mode is train or test")
