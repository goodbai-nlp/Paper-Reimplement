#!/usr/bin/env python
# encoding: utf-8


"""
@version: ??
@author: muyeby
@contact: bxf_hit@163.com
@site: http://muyeby.github.io
@software: PyCharm
@file: CreateBLdict.py
@time: 17-7-13 下午3:25
"""

GoldDictFile = "/home/xfbai/mywork/git/Bilingual-lexicon-survey/UBiLexAT/evaluation/ldc_cedict.txt"
#EavlDictFile = "/home/xfbai/mywork/git/Bilingual-lexicon-survey/UBiLexAT/evaluation/result.1"
EavlDictFile = "result.1"
import argparse

def LoadGoldDict(Dictdir):
    ResDict = {}
    for lines in open(Dictdir):
        items = lines.strip().split('/')
        word = items[0].strip()
        # tmp = ' '.join(items[1:-1])
        # trans = tmp.split(' ')
        trans = items[1:-1]
        if(word not in ResDict):
            ResDict[word]=trans
        else:
            print(word + "already exists!")
    return ResDict

def LoadTestDict(Dictdir):
    ResDict = {}
    for lines in open(Dictdir):
        items = lines.strip().split('\t')
        word = items[0].strip()
        trans = items[1].strip().split(' ')
        if (word not in ResDict):
            ResDict[word] = trans
        else:
            print(word + "already exists!")
    return ResDict

def ComputeAccuracy(GoldDict,TestDict,topk):
    count,correct,flag = 0,0,0
    for key,value in TestDict.items():
        if(key not in GoldDict):
            continue
        goldlist = GoldDict[key]
        flag=0
        for item in goldlist:
            if len(item.split(' '))>1:
                flag=1
#                print item
                break
        if(flag==0):
            count += 1
            for i in range(topk):
                word = value[i]
                if(word in goldlist):
                    correct+=1
                    break
    print(str(correct)+ "/" + str(count)+" % age " + str(correct*1.0/count))

if __name__ == "__main__":
    topk = 1
    parser = argparse.ArgumentParser()
    parser.add_argument('gold',type = str, help='gold dict.')
    parser.add_argument('test',type = str, help='test result')
    args = parser.parse_args()
    GoldDictFile = args.gold
    EavlDictFile = args.test
    GoldDict = LoadGoldDict(GoldDictFile)
    TestDict = LoadTestDict(EavlDictFile)
    ComputeAccuracy(GoldDict,TestDict,topk)
