from pybloom import BloomFilter
from transformers import BertTokenizer
import pandas as pd
import os
import csv
import codecs

'''write index pairs into csv'''
def data_write_csv(file_name, datas):#file_name为写入CSV文件的路径，datas为要写入数据列表
    file_csv = codecs.open(file_name,'w+','utf-8')#追加
    writer = csv.writer(file_csv)
    for data in datas:
        writer.writerow(data)


tokenizerPath = '/home/smz/models/bert-small-finetuned/vocab.txt'

'''dataset preprocessing'''
pathA = os.path.join('.','data','er_magellan','Beer_Origin','beerA.csv')
pathB = os.path.join('.','data','er_magellan','Beer_Origin','beerB.csv')

tempA = pd.read_csv(pathA)
tempB = pd.read_csv(pathB)
datasetA = tempA.iloc[:,0]
datasetB = tempB.iloc[:,0]

for i in range(tempA.shape[1]):
    datasetA = datasetA + ' ' + tempA.iloc[:, i]
for i in range(tempB.shape[1]):
    datasetB = datasetB + ' ' + tempB.iloc[:, i]

tokenizer = BertTokenizer.from_pretrained(tokenizerPath)
'''bf of dataset A'''
f1 = []
bfA = [0]
for i in range(tempA.shape[0]):
    f1.append(BloomFilter(capacity=64, error_rate=0.1))
    token = tokenizer.tokenize(datasetA[i])
    for j in range(token.__len__()):
        f1[i].add(token[j])
    #bfA.append(f1[i+1])
'''bf of dataset B'''
f2 = []
bfB = [0]
for i in range(tempB.shape[0]):
    f2.append(BloomFilter(capacity=64, error_rate=0.1))
    token = tokenizer.tokenize(datasetB[i])
    for j in range(token.__len__()):
        f2[i].add(token[j])
    #bfB.append(f2[i+1])


'''Cartesian product'''
_inter = []
idx = []
for i in range(f1.__len__()):
    for j in range(f2.__len__()):
        inter = f1[i].intersection(f2[j]).bitarray.tolist().count(1)
        if inter >= 30:
            idx.append([i,j])

'''write index to csv'''
data_write_csv('blocked_index.csv',idx)

#inter1 = f[1].intersection(f[2]).bitarray.tolist().count(1)

print(f1.__len__())