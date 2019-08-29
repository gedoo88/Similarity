import math
import numpy as np
import pandas as pd
import scipy
import scipy.sparse
import scipy.sparse.linalg
import copy
import random
import importlib
import sim_learning
import benchmark_methods_cross
import time
import sys
import random
import matplotlib.pyplot as plt
from time import perf_counter



def A_from_edges(data_edges, n_nodes):

        nedges = len(data_edges)
        row_ind = np.zeros(2*nedges, dtype = int)
        col_ind = np.zeros(2*nedges, dtype = int)

        row_ind[:nedges] = data_edges[:,0]
        row_ind[nedges:] = data_edges[:,1]
        col_ind[:nedges] = data_edges[:,1]
        col_ind[nedges:] = data_edges[:,0]

        A_csr = scipy.sparse.csr_matrix((np.ones(2*nedges, dtype = float),(row_ind,col_ind)), shape=(n_nodes,n_nodes))
        return A_csr



def get_prec100(results, test_group):

        in_group= np.empty(shape = 0, dtype = bool)
        for i in np.argsort(-results)[:100]:
                in_group = np.append(in_group, i in test_group)
        return np.mean(in_group)



def get_accuracy(results, test_group):

        in_group= np.empty(shape = 0, dtype = bool)
        for i in np.argsort(-results)[:len(test_group)]:
                in_group = np.append(in_group, i in test_group)
        accuracy = np.mean(in_group)
        return np.mean(in_group)



def get_NDCG(results, test_group):

        x = np.arange(len(test_group))
        IDCG = np.sum(1/np.log(x+2))
        sum_NDCG = 0
        y = np.empty(len(test_group), dtype = int)
        z = np.argsort(-results)
        for i in range(len(test_group)):
                y[i] = np.argwhere(z == test_group[i])
        sum_NDCG = np.sum(1/np.log(y+2))
        NDCG = sum_NDCG/IDCG 
        return NDCG



number_nodes=10312
edges = np.empty([333983,2],dtype=int)
file = open(r'Datasets\blogcat_ungraph.txt', 'r')
row = 0
for line in file:
        edges[row] = np.fromstring(line, sep="\t")
        row += 1
groups = r'Datasets\blogcat_contiguous_large_groups.txt'
A = A_from_edges(edges,number_nodes)



group_sel=np.arange(29)
random.shuffle(group_sel)
n_groups = 29
prec100= np.empty((n_groups,10,4))  
accuracy= np.empty((n_groups,10,4)) 
NDCG= np.empty((n_groups,10,4))         
for i in range(n_groups):
        for j in range(10):
                results = sim_learning.get_sim_score(A, number_nodes, groups, group_sel[i], 1e-02, 1, 16, 10)
                prec100[i,j,0] = get_prec100(results[0],results[1])
                accuracy[i,j,0] = get_accuracy(results[0],results[1])
                NDCG[i,j,0] = get_NDCG(results[0],results[1])
                results = benchmark_methods_cross.get_PPR_score(A, number_nodes, groups, group_sel[i], 1e-02, 1, 16, 10)
                prec100[i,j,1] = get_prec100(results[0],results[1])
                accuracy[i,j,1] = get_accuracy(results[0],results[1])
                NDCGnl[i,j,1] = get_NDCG(results[0],results[1])
                results = benchmark_methods_cross.get_exp_score(A, number_nodes, groups, group_sel[i], 1e-02, 1, 16, 10)
                prec100[i,j,2] = get_prec100(results[0],results[1])
                accuracy[i,j,2] = get_accuracy(results[0],results[1])
                NDCGnl[i,j,2] = get_NDCG(results[0],results[1])
                results = benchmark_methods_cross.get_LLGC_score(A, number_nodes, groups, group_sel[i], 1e-02, 1, 16, 10)
                prec100[i,j,3] = get_prec100(results[0],results[1])
                accuracy[i,j,3] = get_accuracy(results[0],results[1])
                NDCGnl[i,j,3] = get_NDCG(results[0],results[1])
        print(i)

result_prec100 = np.empty((4,n_groups))
for i in range(n_groups):
        mean = np.mean([np.mean(prec100[i,:,1]),np.mean(prec100[i,:,2]),np.mean(prec100[i,:,3]),np.mean(prec100[i,:,4])])
        std = np.std([prec100[i,:,1],prec100[i,:,2],prec100[i,:,3],prec100[i,:,4]])
        result_prec100[0,i] = (np.mean(prec100[i,:,1])-mean)/std
        result_prec100[1,i] = (np.mean(prec100[i,:,2])-mean)/std
        result_prec100[2,i] = (np.mean(prec100[i,:,3])-mean)/std
        result_prec100[3,i] = (np.mean(prec100[i,:,4])-mean)/std
score_prec100 = np.empty(4)
for i in range(4):
        score_prec100[i]=np.mean(result_prec100[i,:])        

result_acc = np.empty((4,n_groups))
for i in range(n_groups):
        mean = np.mean([np.mean(accuracy[i,:,1]),np.mean(accuracy[i,:,2]),np.mean(accuracy[i,:,3]),np.mean(accuracy[i,:,4])])
        std = np.std([accuracy[i,:,1],accuracy[i,:,2],accuracy[i,:,3],accuracy[i,:,4]])
        result_acc[0,i] = (np.mean(accuracy[i,:,1])-mean)/std
        result_acc[1,i] = (np.mean(accuracy[i,:,2])-mean)/std
        result_acc[2,i] = (np.mean(accuracy[i,:,3])-mean)/std
        result_acc[3,i] = (np.mean(accuracy[i,:,4])-mean)/std
score_acc = np.empty(4)
for i in range(4):
        score_acc[i]=np.mean(result_acc[i,:])        

result_NDCG = np.empty((4,n_groups))
for i in range(n_groups):
        mean = np.mean([np.mean(NDCG[i,:,1]),np.mean(NDCG[i,:,2]),np.mean(NDCG[i,:,3]),np.mean(NDCG[i,:,4])])
        std = np.std([NDCG[i,:,1],NDCG[i,:,2],NDCG[i,:,3],NDCG[i,:,4]])
        result_NDCG[0,i] = (np.mean(NDCG[i,:,1])-mean)/std
        result_NDCG[1,i] = (np.mean(NDCG[i,:,2])-mean)/std
        result_NDCG[2,i] = (np.mean(NDCG[i,:,3])-mean)/std
        result_NDCG[3,i] = (np.mean(NDCG[i,:,4])-mean)/std
score_NDCG = np.empty(4)
for i in range(4):
        score_NDCG[i]=np.mean(result_NDCG[i,:])        


simlearn = (score_prec100[0],score_acc[0],score_NDCG[0])
PPR = (score_prec100[1],score_acc[1],score_NDCG[1])
Exp = (score_prec100[2],score_acc[2],score_NDCG[2])
LLGC = (score_prec100[3],score_acc[3],score_NDCG[3])
index = np.arange(3)
bar_width = 0.2

fig,ax = plt.subplots()
plt.xlim(-0.7, 3)
rec1 = plt.bar(index-bar_width, simlearn, width=bar_width, bottom=None, color = 'darkorange', label = 'SimLearn',
edgecolor = 'black', linewidth = 1, linestyle = '--')
rec2 = plt.bar(index, PPR, width=bar_width, bottom=None, color = 'darkblue',label = 'PPR',
edgecolor = 'black', linewidth = 1, linestyle = '-')
rec3 = plt.bar(index+bar_width, Exp, width=bar_width, bottom=None, color = 'darkgreen',label = 'Exp',
edgecolor = 'black', linewidth = 1, linestyle = '-')
rec4 = plt.bar(index+2*bar_width, LLGC, width=bar_width, bottom=None, color = 'darkred',label = 'LLGC',
edgecolor = 'black', linewidth = 1, linestyle = '-')
plt.xticks(index + bar_width/2, ('prec100', 'acc', 'NDCG'))
plt.ylabel('Std. dev. from average results')
plt.grid(axis = 'y', linestyle = '--')
plt.legend(loc = 'bottom left')
plt.show()





