# 文本分类算法
总结一些在学习工作中用过的的文本分类算法，欢迎star！
## ML 基于机器学习的文本分类
这里找到一份垃圾短信数据，机器学习文件夹下为垃圾短信分类。
### 1、基于逻辑回归的文本分类算法
lr_main.py <br />
逻辑回归算法的原理在博客中进行了简单介绍  https://blog.csdn.net/Explorer_Du/article/details/84067510<br />
### 2、基于朴树贝叶斯的文本分类算法
nb_main.py<br />
朴素贝叶斯算法原理在博客中进行了简单介绍  https://blog.csdn.net/Explorer_Du/article/details/85242690<br />
### 3、基于支持向量机的文本分类算法
svm_main.py <br />
## DL 基于深度学习的文本分类
### 1、目录结构
<ul>
    <li>data: 训练数据集文件夹</li>
    <li>data_helper: 数据预处理</li>
    <li>models: 模型结构</li>
    <li>result: 存放结果的文件夹</li>
    <li>util: 训练词向量</li>
    <li>run.py: 训练模型文件</li>
    <li>train_val_test.py: 具体训练验证和测试的代码</li>
</ul>

### 2、数据
使用THUCNews的一个子集进行训练与测试，由<a herf="https://github.com/gaussic/text-classification-cnn-rnn">gaussic</a>提供。<br />
类别如下：
```
体育, 财经, 房产, 家居, 教育, 科技, 时尚, 时政, 游戏, 娱乐
```
百度网盘下载链接在data文件夹下的README。<br />
数据集划分如下：<br />
<ul>
    <li>训练集: 5000*10</li>
    <li>验证集: 500*10</li>
    <li>测试集: 1000*10</li>
</ul>
数据文件：
<ul>
    <li>cnews.train.txt: 训练集(50000条)</li>
    <li>cnews.val.txt: 验证集(5000条)</li>
    <li>cnews.test.txt: 测试集(10000条)</li>
</ul>

### 3、运行
<ol>
    <li>在util文件夹下运行：python3 word2vec.py</li>
    <li>训练模型，运行run.py: python3 run.py --model xxx --mode train</li>
    <li>测试模型，运行run.py: python3 run.py --model xxx --mode test</li>
</ol>
xxx代表训练的模型，有TextCNN、TextRNN、HAN、TextRCNN、DPCNN、TextRNN_Att和Transformer

### 4、 结果

### 5、参考
[1] https://github.com/gaussic/text-classification-cnn-rnn<br />
[2] https://github.com/cjymz886/text-cnn <br />
[3] https://github.com/649453932/Chinese-Text-Classification-Pytorch


