'''困惑度计算LDA模型'''
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import re
import jieba
import gensim
from pprint import pprint
import gensim.corpora as corpora
from gensim.models import CoherenceModel
from engLemmatization import getAllSentenceLemmatization
# from gensim.models import CoherenceModel

# Plotting tools
import pyLDAvis
import pyLDAvis.gensim  # don't skip this
import matplotlib.pyplot as plt
# %matplotlib inline

## 去除文本中的表情字符（只保留中英文和数字）
def clear_character(sentence):
    pattern = re.compile('[^\u4e00-\u9fa5^a-z^A-Z^0-9]')
    line = re.sub(pattern, '', sentence)
    new_sentence = ''.join(line.split())
    return new_sentence

# df = pd.read_excel('../data/sample_1000.xlsx', delimiter="\t", header=None)
dataframe=pd.read_excel('../data/test2.xls',encoding='utf-8',dtype=str)


train_text1=dataframe['purpose']
train_text1=list(train_text1)
train_text2=dataframe['task']
train_text2=list(train_text2)
train_text3=dataframe['therapy']
train_text3=list(train_text3)
train_text=[]
for w in range(len(train_text1)):
    strContent=''
    strContent+=train_text1[w]
    strContent +=' '
    strContent += train_text2[w]
    strContent += ' '
    strContent += train_text3[w]
    train_text.append(strContent)

# train_seg_text = [jieba.lcut(s) for s in train_text]
# print("train_seg_text    ",train_seg_text)
train_seg_text=getAllSentenceLemmatization(train_text)
print("len train_seg_text    ",len(train_seg_text))
print("train_seg_text    ",train_seg_text)
##只是按照空格切分
# train_seg_text=[s.split() for s in train_text]

##按照词干还原
# print("train_seg_text    ",[s.split() for s in train_text]) ##[['句子1词1','句子1词2',...,'句子1词n'],[句子2所有词分开],...,[]]
# print(train_seg_text[:1])

##############################1.1 数据清洗、分词###############################
## 加载停用词
stop_words_path = "stopwords_eng.txt"

def get_stop_words():
    return set([item.strip() for item in open(stop_words_path, 'r',encoding='utf-8').readlines()])

stopwords = get_stop_words()

## 去掉文本中的停用词
def drop_stopwords(line):
    line_clean = []
    for word in line:
        if str(word).lower() in stopwords:
            continue
        line_clean.append(word)
    return line_clean


##计算一致性得分
def compute_coherence_values(dictionary, corpus, texts, limit, start=2, step=3):
    """
    Compute c_v coherence for various number of topics

    Parameters:
    ----------
    dictionary : Gensim dictionary
    corpus : Gensim corpus
    texts : List of input texts
    limit : Max num of topics

    Returns:
    -------
    model_list : List of LDA topic models
    coherence_values : Coherence values corresponding to the LDA model with respective number of topics
    """
    coherence_values = []
    model_list = []
    for num_topics in range(start, limit, step):

        model=gensim.models.ldamodel.LdaModel(corpus=corpus,num_topics=num_topics,id2word=id2word,random_state=100,chunksize=100,update_every=1,alpha='auto',passes=10,per_word_topics=True)
        model_list.append(model)
        coherencemodel = CoherenceModel(model=model, texts=texts, dictionary=dictionary, coherence='c_v')
        coherence_values.append(coherencemodel.get_coherence())

    return model_list, coherence_values


train_st_text = [drop_stopwords(s) for s in train_seg_text]

# 二元、三元模型
bigram = gensim.models.Phrases(train_st_text, min_count=5, threshold=100) # higher threshold fewer phrases.
trigram = gensim.models.Phrases(bigram[train_st_text], threshold=100)

## 将句子打成三元组/二元组的更快方法
bigram_mod = gensim.models.phrases.Phraser(bigram)
trigram_mod = gensim.models.phrases.Phraser(trigram)

def make_bigrams(texts):
    return [bigram_mod[doc] for doc in texts]

def make_trigrams(texts):
    return [trigram_mod[bigram_mod[doc]] for doc in texts]

if __name__ == "__main__":
    data_words_bigrams = make_bigrams(train_st_text)

    #################################1.2 构建词典、语料向量化表示#################################
    data_lemmatized=data_words_bigrams

    id2word = corpora.Dictionary(data_words_bigrams)     # Create Dictionary
    texts = data_words_bigrams                           # Create Corpus
    corpus = [id2word.doc2bow(text) for text in texts]   # Term Document Frequency
    # print('2333    ',corpus)
    print(corpus[:1])
    print([[(id2word[id], freq) for id, freq in cp] for cp in corpus[:1]])
    #####################################1.3 构建 LDA 模型#######################################
    # Build LDA model

    ##分为训练与测试去求perplexity值##
    # p = int(len(corpus) * .8)
    # corpus_train, corpus_test = corpus[:p], corpus[p:]

    lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                               id2word=id2word,
                                               num_topics=5,
                                               random_state=100,
                                               update_every=1,
                                               chunksize=100,
                                               passes=10,
                                               alpha='auto',
                                               per_word_topics=True)
    # Print the Keyword in the 10 topics
    pprint(lda_model.print_topics(num_words=10))
    # doc_lda = lda_model[corpus]

    ###############################1.4 求perplex值######################################

    print('\nPerplexity_log: ', lda_model.log_perplexity(corpus))  # a measure of how good the model is. lower the better.

    coherence_model_lda = CoherenceModel(model=lda_model, texts=data_lemmatized, dictionary=id2word, coherence='c_v')
    coherence_lda = coherence_model_lda.get_coherence()
    print('\nCoherence Score: ', coherence_lda)

    ###############################1.5 找到最好的模型（coherence值最高的模型）######################################
    model_list, coherence_values = compute_coherence_values(dictionary=id2word, corpus=corpus, texts=data_lemmatized, start=2, limit=6, step=2)

    # Show graph
    limit=6; start=2; step=2;
    x = range(start, limit, step)
    plt.plot(x, coherence_values)
    plt.xlabel("Num Topics")
    plt.ylabel("Coherence score")
    # annotate()给折线点设置坐标值
    for a, b in zip(x,coherence_values):
        plt.annotate('%s' % (round(b,3)), xy=(a, b), xytext=(-20, 10),
                     textcoords='offset points')

    plt.legend(("coherence_values"), loc='best')
    plt.show()


    # Print the coherence scores
    for m, cv in zip(x, coherence_values):
        print("Num Topics =", m, " has Coherence Value of", round(cv, 4))

    # Compute Coherence Score
    coherence_model_lda = CoherenceModel(model=lda_model, texts=data_words_bigrams, dictionary=id2word, coherence='c_v')
    coherence_lda = coherence_model_lda.get_coherence()
    print('\nCoherence Score: ', coherence_lda)   # 越高越好

# #################################LDA分类测试，每个文档属于每个类的可能性打印出来#######################################
#   # 测试数据转换
# from gensim import corpora
#
# # test_vec=corpus[0:5]##取少的做实验
# test_vec=corpus
#
# # dictionary= corpora.Dictionary
# # test_vec = [dictionary.doc2bow(doc) for doc in test_data]
# # id2word
# #预测并打印结果
# for i, item in enumerate(test_vec):
#     topic = lda_model.get_document_topics(item)
#     # keys = target.keys()
#     # print('第',i+1,'条记录分类结果:',topic)
#     # print(topic)
#     allClasses=[0,1,2,3,4,5,6,7,8,9]
#     # print('hhh   ',allClasses)
#     allClassesProb={}
#     for j in range(len(topic)):
#         # print(topic[j])
#         # print(topic[j][0],'  ',topic[j][1])###9    0.023712367
#         # print(type(topic[j][0]))
#         if topic[j][0] not in allClassesProb.keys():
#             allClassesProb[topic[j][0]]=topic[j][1]
#     for k in range(len(allClasses)):
#         if allClasses[k] not in allClassesProb.keys():
#             allClassesProb[allClasses[k]]='nan'
#     print(allClassesProb[0],' ',allClassesProb[1],' ',allClassesProb[2],' ',allClassesProb[3],' ',allClassesProb[4],' ',allClassesProb[5],' ',allClassesProb[6],' ',allClassesProb[7],' ',allClassesProb[8],' ',allClassesProb[9])
#
# #############################只打印出每个类最可能的类别####################################
# test_vec=corpus
#
# #预测并打印结果
# for i, item in enumerate(test_vec):
#     topic = lda_model.get_document_topics(item)
#     # print(topic)#[(0, 0.015959902), (1, 0.7344932), (2, 0.013574445), (4, 0.015873251), (6, 0.017165257)]
#     max_cate=0#最可能的类别
#     max_prob=0#最可能的类别对应的概率
#     for j in range(len(topic)):
#         # print(topic[j])
#         # print(topic[j][0],' + ',topic[j][1])###9    0.023712367
#         # print(type([j][0]))
#         # print('float(topic[j][1])=',float(topic[j][1]),'   float(topic[j][1])>max_prob=',float(topic[j][1])>max_prob)
#         if float(topic[j][1])>max_prob:
#             max_cate=topic[j][0]
#             max_prob=float(topic[j][1])
#     print(max_cate)