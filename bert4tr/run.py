# Author: dgm
# Description: 运行程序
# Date: 2020-11-06
import time
import torch
import argparse
import numpy as np
from importlib import import_module
from bert4tr.train_val_test import train
from bert4tr.utils import build_dataset, build_iterator, get_time_dif

parser = argparse.ArgumentParser(description='Chinese Text Classification')
parser.add_argument('--model', type=str, required=True, help='choose a model: Bert')
args = parser.parse_args()


if __name__ == '__main__':
    model_name = args.model  # bert
    x = import_module('models.' + model_name)
    config = x.Config()
    np.random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    torch.backends.cudnn.deterministic = True  # 保证每次结果一样

    start_time = time.time()
    print("Loading data...")
    train_data, dev_data, test_data = build_dataset(config)
    train_iter = build_iterator(train_data, config)
    dev_iter = build_iterator(dev_data, config)
    test_iter = build_iterator(test_data, config)
    time_dif = get_time_dif(start_time)
    print("Time usage:", time_dif)

    # train
    model = x.BERTModel(config).to(config.device)
    train(config, model, train_iter, dev_iter, test_iter)