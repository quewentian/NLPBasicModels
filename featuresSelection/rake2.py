'''抽取单篇文档的key word'''
import re
import operator
import argparse
import codecs
import os

def isNumber(s):
    try:
        float(s) if '.' in s else int(s)
        return True
    except ValueError:
        return False


class Rake:

    def __init__(self, inputFilePath, stopwordsFilePath, outputFilePath, minPhraseChar, maxPhraseLength):
        self.outputFilePath = outputFilePath
        self.minPhraseChar = minPhraseChar
        self.maxPhraseLength = maxPhraseLength
        # read documents
        self.docs = []      ###self.docs一篇文章的内容
        for document in codecs.open(inputFilePath, 'r', 'utf-8',errors='ignore'):###document是一句话
            print('doc111 ',document)
            self.docs.append(document)
        # read stopwords
        stopwords = []
        for word in codecs.open(stopwordsFilePath, 'r', 'utf-8'):
            stopwords.append(word.strip())
        stopwordsRegex = []
        for word in stopwords:
            regex = r'\b' + word + r'(?![\w-])'
            stopwordsRegex.append(regex)
        self.stopwordsPattern = re.compile('|'.join(stopwordsRegex), re.IGNORECASE)

    def separateWords(self, text):###把句子打散成短语的形式
        splitter = re.compile('[^a-zA-Z0-9_\\+\\-/]')
        words = []
        for word in splitter.split(text):
            word = word.strip().lower()
            # leave numbers in phrase, but don't count as words, since they tend to invalidate scores of their phrases
            if len(word) > 0 and word != '' and not isNumber(word):
                words.append(word)
        return words

    def calculatePhraseScore(self, phrases):
        # calculate wordFrequency and wordDegree
        wordFrequency = {}###计算一个单词在文章中出现的次数，分母
        wordDegree = {}####单词的度，两个词的短语，度相当于1，词有一个邻居！！！！！！！在这里继续改动，把在一个sentence里计算改成在所有领域文本中计算
        for phrase in phrases:
            # print('phrase   ',phrase)###phrase:切分后的短语，strategic planning
            wordList = self.separateWords(phrase)
            print('wordlist   ',wordList)###wordlist：短语变成word的list，['strategic', 'planning']
            wordListLength = len(wordList)###wordListLength： 短语的长度
            wordListDegree = wordListLength - 1###wordListDegree等于短语长度减去1
            print('wordListDegree   ',wordListDegree)
            for word in wordList:
                wordFrequency.setdefault(word, 0)
                wordFrequency[word] += 1
                wordDegree.setdefault(word, 0)
                wordDegree[word] += wordListDegree
        for item in wordFrequency:
            wordDegree[item] = wordDegree[item] + wordFrequency[item]
            print('wordDegree[item]   ',wordDegree[item])

        # calculate wordScore = wordDegree(w)/wordFrequency(w)
        wordScore = {}
        for item in wordFrequency:
            wordScore.setdefault(item, 0)
            wordScore[item] = wordDegree[item] * 1.0 / wordFrequency[item]

        # calculate phraseScore
        phraseScore = {}
        for phrase in phrases:
            phraseScore.setdefault(phrase, 0)
            wordList = self.separateWords(phrase)
            candidateScore = 0
            for word in wordList:####wordlist：短语变成word的list，['strategic', 'planning']
                candidateScore += wordScore[word]##短语的score等于每个单词的score加和
            phraseScore[phrase] = candidateScore###phraseScore：每个短语的得分
        return phraseScore

    def execute(self):
        # file = codecs.open(self.outputFilePath, 'w', 'utf-8')
        # print('doc111   ',self.docs)
        allKeyWords={}
        for document in self.docs:###一篇文章的每句话循环
            print('docu333   ',document)
            # split a document into sentences
            sentenceDelimiters = re.compile(u'[.!?,;:\t\\\\"\\(\\)\\\'\u2019\u2013]|\\s\\-\\s')
            sentences = sentenceDelimiters.split(document)
            # generate all valid phrases
            phrases = []
            for s in sentences:
                tmp = re.sub(self.stopwordsPattern, '|', s.strip())
                phrasesOfSentence = tmp.split("|")
                for phrase in phrasesOfSentence:
                    phrase = phrase.strip().lower()
                    if phrase != "" and len(phrase) >= self.minPhraseChar and len(
                            phrase.split()) <= self.maxPhraseLength:
                        phrases.append(phrase)

            # calculate phrase score
            phraseScore = self.calculatePhraseScore(phrases)
            if phraseScore!={}:
                print(phraseScore,'  333')
                keywords = sorted(phraseScore.items(), key=operator.itemgetter(1), reverse=True)
                print(type(keywords))
                allKeyWords.update(keywords)

        print(allKeyWords,'   111')
        print('here      ',sorted(allKeyWords.items(), key=lambda item:item[1], reverse=True))
        sortDict=sorted(allKeyWords.items(), key=lambda item: item[1], reverse=True)
        if len(sortDict)>=50:
            sortDictCut = dict(sortDict[0:49])
        else:
            sortDictCut = dict(sortDict)
        # sortDictCut=dict(sortDict[0:int(len(sortDict) / 3)])单篇短文本考虑抽取尽可能的的关键词，50个
        file = codecs.open(self.outputFilePath, 'w', 'utf-8')
        for i in sortDictCut.keys():
            outline=''
            outline+=i
            outline+=':'
            outline+=str(sortDictCut[i])
            file.write(outline + "\n")
        file.close()

file_dir='new-experiment-LDA/5AbstractsGroup-test'
dir_list=os.listdir(file_dir)
for i in range(len(dir_list)):
    second_dir=os.path.join(file_dir,dir_list[i])
    dir_list2=os.listdir(second_dir)
    for j in range(len(dir_list2)):
        path=os.path.join(second_dir,dir_list2[j])
        outpath=path.replace('5AbstractsGroup-test','keyword/5AbstractsGroup-test-new')
        # print(outpath,'   out222')

        rake = Rake(path,'data/stoplists/SmartStoplist.txt',outpath,1,2)
        rake.execute()