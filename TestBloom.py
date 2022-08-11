from pybloom import BloomFilter
from transformers import BertTokenizer
import pandas as pd
import os


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
f1 = [0]
bfA = [0]
for i in range(tempA.shape[0]):
    f1.append(BloomFilter(capacity=64, error_rate=0.1))
    token = tokenizer.tokenize(datasetA[i])
    for j in range(token.__len__()):
        f1[i + 1].add(token[j])
    bfA.append(f1[i+1])
'''bf of dataset B'''
f2 = [0]
bfB = [0]
for i in range(tempB.shape[0]):
    f2.append(BloomFilter(capacity=64, error_rate=0.1))
    token = tokenizer.tokenize(datasetB[i])
    for j in range(token.__len__()):
        f2[i + 1].add(token[j])
    bfB.append(f2[i+1])


'''Cartesian product'''



# l1 = f[1].bitarray.tolist()
# l2 = f[2].bitarray.tolist()
#
# num1 = l1.count(1)
# num2 = l2.count(1)

#inter1 = f[1].intersection(f[2]).bitarray.tolist().count(1)

#print(f[1].__len__())