# 文本分类算法
总结一些在学习工作中用过的的文本分类算法，本地容易丢失，所以放到github上，喜欢的就star哦！
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
### 1、基于CNNh和RNN的文本分类算法
TF_CNN_RNN目录
#### (1)目录结构
&nbsp;&nbsp;&nbsp;&nbsp;config--配置文件<br />
&nbsp;&nbsp;&nbsp;&nbsp;data--数据存放路径<br />
&nbsp;&nbsp;&nbsp;&nbsp;data_helper--数据处理<br />
&nbsp;&nbsp;&nbsp;&nbsp;model--模型结构<br />
&nbsp;&nbsp;&nbsp;&nbsp;result--存放结果<br />
&nbsp;&nbsp;&nbsp;&nbsp;utils--工具（训练词向量）<br />
&nbsp;&nbsp;&nbsp;&nbsp;predict.py--模型预测<br />
&nbsp;&nbsp;&nbsp;&nbsp;server.py--启一个服务<br />
&nbsp;&nbsp;&nbsp;&nbsp;train_cnn.py--训练CNN模型<br />
&nbsp;&nbsp;&nbsp;&nbsp;train_rnn.py--训练RNN模型<br />
#### （2）数据
使用THUCNews的一个子集进行训练与测试，数据来源于[gaussic](https://github.com/gaussic/text-classification-cnn-rnn)大佬。<br />
一共10个分类,每个分类6500条数据:<br />
```
体育, 财经, 房产, 家居, 教育, 科技, 时尚, 时政, 游戏, 娱乐
```
数据集合划分:
<ul>
    <li>训练集:5000*10</li>
    <li>验证集:500*10</li>
    <li>测试集:1000*10</li>
</ul>
需要的同学可以去gaussic的github上自行下载。

#### (3)运行
<ul>
    <li>step1:在utils下，python3 word2vec.py</li>
    <li>step2:训练CNN，python3 train_cnn.py</li>
    <li>step3:训练RNN，python3 train_rnn.py</li>  
    <li>step4:预测模型，python3 predict.py</li>
    <li>step5:启动服务，python3 server.py</li>
</ul>

#### (4)参考
[1] https://github.com/gaussic/text-classification-cnn-rnn<br />
[2] https://github.com/cjymz886/text-cnn

