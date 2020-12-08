# AUthor: duguiming
# Description: 基础的配置项目
# Date: 2020-03-31
import os

base_dir = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))


class BaseConfig(object):
    # 训练集、验证集和测试集地址
    train_dir = os.path.join(base_dir, 'data/cnews/train.txt')
    val_dir = os.path.join(base_dir, 'data/cnews/dev.txt')
    test_dir = os.path.join(base_dir, 'data/cnews/test.txt')

    # 数据类别
    class_dir = os.path.join(base_dir, 'data/cnews/class.txt')
    class_list = [x.strip() for x in open(class_dir, 'r', encoding='utf-8').readlines()]
    num_classes = len(class_list)