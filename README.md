# 文本分类算法
总结一些在学习工作中用过的的文本分类算法，欢迎star！

## ML 基于机器学习的文本分类

### 1、目录结构
<ul>
    <li>data : 垃圾短信数据</li>
    <li>lr_main.py : 基于逻辑回归的文本分类</li>
    <li>nb_main.py: 基于朴素贝叶斯的文本分类</li>
    <li>svm_main.py: 基于支持向量机的文本分类</li>
</ul>

### 2、数据
这里找到一份垃圾短信数据，data文件夹下为垃圾短信分类。

### 3、 运行
<ol>
    <li>python3 lr_main.py</li>
    <li>python3 nb_main.py</li>
    <li>python3 svm_main.py</li>
</ol>

### 4、博客
[1] https://blog.csdn.net/Explorer_Du/article/details/84067510<br />
[2] https://blog.csdn.net/Explorer_Du/article/details/85242690<br />

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
使用THUCNews的一个子集进行训练与测试，由<a href="https://github.com/gaussic/text-classification-cnn-rnn">gaussic</a>提供。<br />
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

### 4、 效果
<table>
    <tr>
        <td>模型</td>
        <td>val准确率</td>
        <td>test准确率</td>
        <td>备注</td>
    </tr>
    <tr>
        <td>TextCNN</td>
        <td>0.931</td>
        <td>0.945</td>
        <td>step为2200时候，提前终止了</td>
    </tr>
    <tr>
        <td>TextRNN</td>
        <td>0.946</td>
        <td>0.97</td>
        <td>训练的速度相对TextCNN较慢</td>
    </tr>
    <tr>
        <td>TextRCNN</td>
        <td>0.922</td>
        <td>0.964</td>
        <td>GRU+pooling</td>
    </tr>
    <tr>
        <td>TextRNN_Att</td>
        <td>0.935</td>
        <td>0.962</td>
        <td>内存不够可以调小batch_size</td>
    </tr>
    <tr>
        <td>DPCNN</td>
        <td>0.934</td>
        <td>0.955</td>
        <td>训练的速度较快</td>
    </tr>
    <tr>
        <td>HAN</td>
        <td>0.913</td>
        <td>0.95</td>
        <td>采用GRU</td>
    </tr>
    <tr>
        <td>Transformer</td>
        <td>0.909</td>
        <td>0.92</td>
        <td>效果最差...</td>
    </tr>
</table>

### 5、更新
<ul>
    <li>2020年11月6日，增加基于pytorch版bert文本分类实现</li>
    <li>2020年3月31日，增加DL（深度学习）文本分类实现</li>
    <li>2019年7月5日，增加机器学习垃圾短信分类实现</li> 
</ul>

### 六、参考
[1] https://github.com/gaussic/text-classification-cnn-rnn<br />
[2] https://github.com/cjymz886/text-cnn <br />
[3] https://github.com/649453932/Chinese-Text-Classification-Pytorch


