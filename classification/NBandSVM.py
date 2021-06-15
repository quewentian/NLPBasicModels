from sklearn import model_selection, preprocessing, linear_model, naive_bayes, metrics, svm
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn import decomposition, ensemble

# import pandas, xgboost, numpy, textblob, string
# from keras.preprocessing import text, sequence
# from keras import layers, models, optimizers
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.metrics import classification_report
# -*- encoding=utf-8 -*-

# 导入包
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import BaseNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
# import jieba
import sklearn
from sklearn.metrics import classification_report
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
# import jieba.analyse

# 设置不以科学计数法输出
import numpy as np
import os
import codecs

np.set_printoptions(suppress=True)


# # 载入自定义词典
# jieba.load_userdict(r'F:\文本标签\文本反垃圾\恶意推广文本分类\第一类赛事主体标签.txt')
# jieba.load_userdict(r'F:\文本标签\文本反垃圾\恶意推广文本分类\第二类网络主体标签.txt')
#
# # 载入自定义停止词
# jieba.analyse.set_stop_words(r'F:\文本标签\文本反垃圾\恶意推广文本分类\stopwords.txt')


# 数据预处理
def data_preprocessing(file):
    """
    :param df: 数据集，类型为数据框
    :return: 返回切词后的数据集
    """
    file_dir=os.listdir(file)
    text_list=[]
    category_list=[]

    for i in range(len(file_dir)):
        category=file_dir[i]
        second_dir=os.path.join(file,file_dir[i])
        # print(second_dir)
        second_dir_list=os.listdir(second_dir)
        for j in range(len(second_dir_list)):
            # print(second_dir_list[j])
            path=os.path.join(second_dir,second_dir_list[j])
            # print(path)
            text=''
            label=category
            with codecs.open(path,'r',encoding='utf-8',errors='ignore') as f:
                s=f.read()
                # print(type(s))##string, type
                text_list.append(s)
                category_list.append(label)
    return text_list,category_list



# 拆分训练数据集和测试集
def data_spilt_train_test(news_data,news_target):
    """
    :param news_data: 分词后的数据集
    :param news_target: 目标变量
    :return: 拆分后的训练集和测试集
    """
    # 分割数据集
    x_train, x_test, y_train, y_test = train_test_split(news_data, news_target, test_size=0.333,random_state=1000)
    return x_train, x_test, y_train, y_test


# 模型训练
def model_train(x_train, x_test, y_train, y_test):

    # # 进行tfidf特征抽取
    # tf = TfidfVectorizer()
    # x_train = tf.fit_transform(x_train)
    # x_test = tf.transform(x_test)

    # 创建一个向量计数器对象
    count_vect = CountVectorizer(analyzer='word', token_pattern=r'\w{1,}')
    # count_vect = CountVectorizer(analyzer='word', token_pattern=r'\w{1,}')
    # count_vect.fit(trainDF['text'])
    # 使用向量计数器对象转换训练集和验证集
    x_train = count_vect.fit_transform(x_train)
    x_test = count_vect.transform(x_test)

    print('x_test    ',x_test)
    # print('y_test    ',y_test)

    # 通过朴素贝叶斯进行预测(拉普拉斯平滑系数为设置为1)
    # mlb = MultinomialNB( alpha=1)
    # mlb = BaseNB()
    # mlb=GaussianNB()
    # mlb=BernoulliNB()


    mlb=SVC(C=0.99, kernel = 'linear')
    # mlb=KNeighborsClassifier()
    mlb.fit(x_train, y_train)
    predict = mlb.predict(x_test)

    # print(predict)

    count = 0  # 统计预测正确的结果个数
    for left, right in zip(predict, y_test):
        if left == right:
            count += 1
    print(count / len(y_test),'   222')

    f1 = f1_score(y_test, predict, average='macro')
    p = precision_score(y_test, predict, average='macro')
    r = recall_score(y_test, predict, average='macro')

    # t = classification_report(y_true, y_pred, target_names=['business', 'politics', 'tech','entertainment','sport'],digits=4)
    # t = classification_report(y_test, predict, target_names=['accounts', 'biology', 'geography', 'physics'], digits=4)
    t = classification_report(y_test, predict, digits=4)

    print(t)



    # 训练集上的评测结果
    # accuracy, auc = evaluate(mlb, x_train, y_train)
    # print("训练集正确率：%.4f%%\n" % (accuracy * 100))
    # print("训练集AUC值：%.6f\n" % (auc))
    #
    # # 测试集上的评测结果
    # accuracy, auc = evaluate(mlb, x_test, y_test)
    # print("测试集正确率：%.4f%%\n" % (accuracy * 100))
    # print("测试AUC值：%.6f\n" % (auc))

    # y_predict = mlb.predict(x_test)
    # print(classification_report(y_test, y_predict, target_names=['0', '1']))
    return mlb
    # return mlb,tf



# 模型评估
def evaluate(model, X, y):
    """评估数据集，并返回评估结果，包括：正确率、AUC值
    """
    accuracy = model.score(X, y)
    fpr, tpr, thresholds = sklearn.metrics.roc_curve(y, model.predict_proba(X)[:, 1], pos_label=1)
    return accuracy, sklearn.metrics.auc(fpr, tpr)


# 模型预测
# # def model_predict(text,model,tf):
# #     """
# #     :param text: 单个文本
# #     :param model: 朴素贝叶斯模型
# #     :param tf: 向量器
# #     :return: 返回预测概率和预测类别
# #     """
# #     text1=[" ".join(jieba.cut(text))]
#
#
#     # 进行tfidf特征抽取
#     text2 = tf.transform(text1)
#
#     predict_type = model.predict(text2)[0]
#
#     predict_prob = model.predict_proba(text2)
#
#     prob_0 = predict_prob[0][0]
#     prob_1 = predict_prob[0][1]
#
#     if predict_type == 1:
#         result_prob = round(prob_1, 3)
#     else:
#         result_prob = round(prob_0, 3)
#
#     return predict_type, result_prob





if __name__ == '__main__':


    # 读取数据
    train_file = "newExperiment\\5AbstractsGroup-train"
    test_file = "newExperiment\\5AbstractsGroup-test"

    # data,target=data_preprocessing(file)
    x_train,y_train=data_preprocessing(train_file)
    x_test,y_test=data_preprocessing(test_file)
    # print(len(x_train))
    # print(len(y_train))
    # print(len(y_test))

    # x_train, x_test, y_train, y_test=data_spilt_train_test(data,target)
    mlb,tf=model_train(x_train, x_test, y_train, y_test)
    # mlb= model_train(x_train, x_test, y_train, y_test)
    # mlb=KNeighborsClassifier()


    # # 预测单文本
    # text="既然选择相信，那就等他涅槃重生吧"
    # predict_type, predict_prob=model_predict(text,mlb,tf)
    # print(predict_type)
    # print(predict_prob)