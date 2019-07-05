# Author:duguiming
# Description:基于支持向量机的垃圾短信识别
# Date:2019-07-05
import jieba
import sklearn
from sklearn.svm import SVC
from sklearn.feature_extraction.text import TfidfVectorizer


def read_data(data_path):
    """
    读取数据
    :param data_path: 数据存放路径
    :return: 读取到的数据
    """
    with open(data_path, 'r', encoding='utf-8') as f:
        data = f.readlines()
    return data


def cut_words(data, stopwords, test_size=0.2):
    """
    分词、去停用词并将数据集分成训练集和测试集
    :param data: 文本数据
    :param stopwords: 停用词
    :param test_size: 测试集和训练集的划分比例
    :return: 测试集和训练集
    """
    stop_words = list()
    for word in stopwords:
        stop_words.append(word[:-1])
    y = list()
    text_list = list()
    for line in data:
        label, text = line.split('\t', 1)
        cut_text = [word for word in jieba.cut(text) if word not in stop_words]
        if cut_text == '':
            continue
        else:
            text_list.append(' '.join(cut_text))
            y.append(int(label))
    return sklearn.model_selection.train_test_split(text_list, y, test_size=test_size, random_state=1028)


def calculate_tfidf(X_train, X_test):
    """
    计算文本的tf-idf
    :param X_train: 训练集
    :param X_test: 测试集
    :return: 返回的是文本的tf-idf特征
    """
    vectorizer = TfidfVectorizer()
    vectorizer.fit_transform(X_train)
    X_train_tfidf = vectorizer.transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    return X_train_tfidf, X_test_tfidf, vectorizer


def evaluate(model, X, y):
    """
    模型评估
    :param model: 训练好的模型
    :param X: 测试集
    :param y: 测试集标签
    :return: 正确率和auc值
    """
    accuracy = model.score(X, y)
    a = model.predict_proba(X)[:, 1]
    fpr, tpr, thresholds = sklearn.metrics.roc_curve(y, model.predict_proba(X)[:, 1], pos_label=1)
    return accuracy, sklearn.metrics.auc(fpr, tpr)


if __name__ == "__main__":
    # step1 读取数据（文本和停用词）
    data_path = "./data/train.txt"
    stopwords_path = "./data/stopwords.txt"
    data = read_data(data_path)
    stopwords = read_data(stopwords_path)

    # step2 分词、分为训练集和测试集
    X_train, X_test, y_train, y_test = cut_words(data, stopwords, test_size=0.2)

    # step3 提取特征参数（tf-idf）
    X_train_tfidf, X_test_tfidf, tfidf_model = calculate_tfidf(X_train, X_test)

    # step4 训练模型
    svm = SVC(C=1.0, probability=True)
    svm.fit(X_train_tfidf, y_train)

    # step5 模型评估
    accuracy, auc = evaluate(svm, X_train_tfidf, y_train)
    print("训练集正确率：%.4f%%\n" % (accuracy * 100))
    print("训练集AUC值：%.6f\n" % auc)

    accuracy, auc = evaluate(svm, X_test_tfidf, y_test)
    print("测试集正确率：%.4f%%\n" % (accuracy * 100))
    print("测试AUC值：%.6f\n" % auc)
