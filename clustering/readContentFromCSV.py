#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2018-05-05 22:15:13
# @Author  : JackPI (1129501586@qq.com)
# @Link    : https://blog.csdn.net/meiqi0538

import jieba

# seg_list = jieba.cut("我来到北京清华大学", cut_all=True)
# strCutList=" ".join(seg_list)
# print("全模式: " ,strCutList)  # 全模式  我/ 来到/ 北京/ 清华/ 清华大学/ 华大/ 大学
# # print(type(strCutList))
#
# seg_list = jieba.cut("我来到北京清华大学", cut_all=False)
# print("精准模式: " + "/ ".join(seg_list))  # 精确模式  我/ 来到/ 北京/ 清华大学
#
# seg_list = jieba.cut("他来到了网易杭研大厦")  # 默认是精确模式
# print(", ".join(seg_list))

import pandas as pd
import string


def jiebaCutList(inlist):
    resultList=[]
    for i in range(len(inlist)):
        seg_list = jieba.cut(inlist[i], cut_all=False)###精准模型中文分词
        strCutList = " ".join(seg_list)##报错AttributeError: 'float' object has no attribute 'decode'，估计是因为cut后有float数字
        resultList.append(strCutList)   ###解决：pd.read_excel('../data/sample_1000.xlsx',encoding='utf-8',dtype=str)，加dtype=str
    # print(resultList)
    return resultList

def cutKeywordsList(inlist):
    resultList=[]
    for i in range(len(inlist)):
        if str(inlist[i])!='NULL' and str(inlist[i])!='nan':
            # print(inlist[i])
            # replaceString=inlist[i].replace(";"," ")
            # replaceString = inlist[i].replace("；", " ")
            # replaceString = inlist[i].replace(";", " ")
            # resultList.append(replaceString)
            replaceString = inlist[i]
            # print(string.punctuation)
            # print(type(string.punctuation))###str
            chinesePunct='；、（）'
            allPunc=string.punctuation+chinesePunct
            # allPunc+='、'
            # print(string.punctuation)
            for c in allPunc:
                # print(c)
                replaceString=replaceString.replace(str(c),' ')
            resultList.append(replaceString)
        else:
            # print(inlist[i])
            resultList.append('')
    # print(resultList)
    return resultList

def concateLists(list1,list2,list3):
    combineList=[]
    for i in range(len(list1)):
        strAll=''
        if list1[i]!='' and list1[i]!='nan':
            strAll+=list1[i]
        if list2[i] != '' and list2[i] != 'nan':
            strAll +=' '
            strAll += list2[i]
        if list3[i]!='' and list3[i]!='nan':
            strAll +=' '
            strAll+=list3[i]
        combineList.append(strAll)
    return combineList

def concateLists2(list1,list2):
    combineList=[]
    for i in range(len(list1)):
        strAll=''
        if list1[i]!='' and list1[i]!='nan':
            strAll+=list1[i]
        if list2[i] != '' and list2[i] != 'nan':
            strAll +=' '
            strAll += list2[i]
        # if list3[i]!='' and list3[i]!='nan':
        #     strAll +=' '
        #     strAll+=list3[i]
        combineList.append(strAll)
    return combineList


#使用pandas读取excel文件（3个字段都用）
def getCombineList():
    # xlsx_file=pd.ExcelFile('../data/sample_1000.xlsx')
    # dataframe=pd.read_excel('../data/sample_1000.xlsx',encoding='utf-8',dtype=str)

    dataframe = pd.read_excel('../data/test.xlsx', encoding='utf-8', dtype=str)
    # xlsx_file.sheet_names#显示出读入excel文件中的表名字
    F_title=dataframe['F_Title']##读取某一列的数据，用标题去读取
    F_Keyword=dataframe['F_Keyword']
    F_Abstract=dataframe['F_Abstract']
    F_title_list=list(F_title)###将object的对象转型成list
    F_Keyword_list=list(F_Keyword)
    F_Abstract_list=list(F_Abstract)
    # print(F_title_list)
    F_title_cut_list=jiebaCutList(F_title_list)
    # print('aaa  ',len(F_title_cut_list))
    F_Abstract_cut_list=jiebaCutList(F_Abstract_list)
    # print('bbb ',len(F_Abstract_cut_list))
    F_Keyword_cut_list=cutKeywordsList(F_Keyword_list)
    # print('ccc ',len(F_Keyword_cut_list))##3个列都是等长的
    combineList=concateLists(F_title_cut_list,F_Abstract_cut_list,F_Keyword_cut_list)
    # print(combineList)
    return combineList
# print(len(combineList))###1000
###把上述3个内容拼接起来，注意有''的和'nan'的

#使用pandas读取excel文件（3个字段都用）——中文
def getCombineList2():
    # xlsx_file=pd.ExcelFile('../data/sample_1000.xlsx')
    dataframe=pd.read_excel('../data/sample_1000.xlsx',encoding='utf-8',dtype=str)
    # xlsx_file.sheet_names#显示出读入excel文件中的表名字
    F_title=dataframe['F_Title']##读取某一列的数据，用标题去读取
    F_Keyword=dataframe['F_Keyword']
    # F_Abstract=dataframe['F_Abstract']
    F_title_list=list(F_title)###将object的对象转型成list
    F_Keyword_list=list(F_Keyword)
    # F_Abstract_list=list(F_Abstract)
    # print(F_title_list)
    F_title_cut_list=jiebaCutList(F_title_list)
    # print('aaa  ',len(F_title_cut_list))
    # F_Abstract_cut_list=jiebaCutList(F_Abstract_list)
    # print('bbb ',len(F_Abstract_cut_list))
    F_Keyword_cut_list=cutKeywordsList(F_Keyword_list)
    # print('ccc ',len(F_Keyword_cut_list))##3个列都是等长的
    combineList=concateLists2(F_title_cut_list,F_Keyword_cut_list)
    # print(combineList)
    return combineList

#使用pandas读取excel文件（3个字段都用）——英文
def getCombineList_eng():
    # xlsx_file=pd.ExcelFile('../data/sample_1000.xlsx')
    dataframe=pd.read_excel('../data/test.xls',encoding='utf-8',dtype=str)
    # xlsx_file.sheet_names#显示出读入excel文件中的表名字
    F_title=dataframe['purpose']##读取某一列的数据，用标题去读取
    # F_Keyword=dataframe['F_Keyword']
    # F_Abstract=dataframe['F_Abstract']
    F_title_list=list(F_title)###将object的对象转型成list

    # print(combineList)
    return F_title_list

#excel文件的写出
#data.to_excel("abc.xlsx",sheet_name="abc",index=False,header=True)  #该条语句会运行失败，原因在于写入的对象是np数组而不是DataFrame对象,只有DataFrame对象才能使用to_excel方法。

# DataFrame(data).to_excel("abc.xlsx",sheet_name="123",index=False,header=True)

#excel文件和pandas的交互读写，主要使用到pandas中的两个函数,一个是pd.ExcelFile函数,一个是to_excel函数


