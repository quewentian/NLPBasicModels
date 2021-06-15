'''使用kmeans和tf-idf去文本聚类'''

# -*- coding: utf-8 -*-
import codecs
from sklearn import feature_extraction
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
# from readContentFromCSV import getCombineList
from readContentFromCSV import getCombineList_eng

##从TXT读取list
# corpus = []
# token_path = "../data/1.txt"
# with open(token_path, 'r') as t:
#     for line in t.readlines():
#         corpus.append(line.strip())
# corpus
# print(corpus)

##从xlsx里直接读取list
corpus=getCombineList_eng()


# 词频矩阵：矩阵元素a[i][j] 表示j词在i类文本下的词频
vectorizer = CountVectorizer()
# 统计每个词语的tf-idf权值
transformer = TfidfTransformer()
freq_word_matrix = vectorizer.fit_transform(corpus)
#获取词袋模型中的所有词语
word = vectorizer.get_feature_names()
# word
print(word)

tfidf = transformer.fit_transform(freq_word_matrix)
# 元素w[i][j]表示j词在i类文本中的tf-idf权重
weight = tfidf.toarray()###weight：每个词的tf-idf值
# weight
print(weight.shape)###(24, 31),行是代表有几篇文档，列代表一篇文档有多少词


### 将每行的（也是每篇文章）文本里的word，转化为tf-idf值
resName = "check.txt"
result = codecs.open(resName, 'w', 'utf-8')
for j in range(len(word)):
    result.write(word[j] + ' ')
result.write('\r\n\r\n')
#打印每类文本的tf-idf词语权重，第一个for遍历所有文本，第二个for便利某一类文本下的词语权重
for i in range(len(weight)):

    for j in range(len(word)):
        result.write(str(weight[i][j]) + ' ')
    result.write('\r\n\r\n')
result.close()

print('weight:   ',weight)
# K-means聚类
print('Start K-means:')
from sklearn.cluster import KMeans
clf = KMeans(n_clusters=5)
s = clf.fit(weight)
# print("clf.fit   ",s)
#20个中心点
print(clf.cluster_centers_)
#每个样本所属的簇
print("clf.labels_   ",clf.labels_)
i = 1
while i <= len(clf.labels_):
    print(i,': ', clf.labels_[i-1])
    i = i + 1
#用来评估簇的个数是否合适，距离越小说明簇分的越好，选取临界点的簇个数
print(clf.inertia_)



from labelText import LabelText
label = clf.labels_
ori_path = "../data/test.txt"
labelAndText = LabelText(label, ori_path)
labelAndText.arrangeLabelTextExcel(write=True)
# labelAndText.sortByLabel(write=True)