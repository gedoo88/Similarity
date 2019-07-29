import math
import numpy as np
import pandas as pd
import os
import scipy
import scipy.sparse
import scipy.sparse.linalg
import copy
import random
import importlib
import sim_learning
import benchmark_methods
import time
from time import perf_counter
importlib.reload(sim_learning)
importlib.reload(benchmark_methods)


#number of nodes -> blogcat: 10312, dblp: 317080
#number of edges -> blogcat: 333983, dblp: 1049866

number_nodes=10312
edges = np.empty([333983,2],dtype=int)
file = open(r'C:\Users\Tim2\My Documents\GitHub\Similarity\blogcat_ungraph.txt', 'r')
#file = open(r'C:\Users\Tim2\My Documents\GitHub\Similarity\dblp_contiguous_ungraph.txt', 'r')
row = 0
for line in file:
        edges[row] = np.fromstring(line, sep="\t")
        row += 1

groups = r'C:\Users\Tim2\My Documents\GitHub\Similarity\blogcat_contiguous_large_groups.txt'
#groups = r'C:\Users\Tim2\My Documents\GitHub\Similarity\dblp_contiguous_large_groups.txt'



def get_sim_results(edges_data, n_nodes, groups, group_number, coef_regu, k, n_split):

        sim_pred = sim_learning.get_similarity_score(edges_data, n_nodes, groups, group_number, coef_regu, k, n_split)
        train_group = sim_pred[1]
        sim_train_0 = copy.deepcopy(sim_pred[0])
        sim_train_0[train_group]=-np.inf
        return sim_train_0, sim_pred[2], train_group, sim_pred[4]



def get_PPR_results(edges_data, n_nodes, groups, group_number, alpha, taylor_order, n_split):

        sim_PPR = benchmark_methods.get_PPR_prediction(edges_data, n_nodes, groups, group_number, alpha, taylor_order, n_split)
        train_group = sim_PPR[1]
        sim_train_0 = copy.deepcopy(sim_PPR[0])
        sim_train_0[train_group]=-np.inf
        return sim_train_0, sim_PPR[2], train_group



def get_exp_results(edges_data, n_nodes, groups, group_number, alpha, taylor_order, n_split):

        sim_exp = benchmark_methods.get_exp_prediction(edges_data, n_nodes, groups, group_number, alpha, taylor_order, n_split)
        train_group = sim_exp[1]
        sim_train_0 = copy.deepcopy(sim_exp[0])
        sim_train_0[train_group]=-np.inf
        return sim_train_0, sim_exp[2], train_group



def get_LLGC_results(edges_data, n_nodes, groups, group_number, alpha, taylor_order, n_split):

        sim_LLGC = benchmark_methods.get_LLGC_prediction(edges_data, n_nodes, groups, group_number, alpha, taylor_order, n_split)
        train_group = sim_LLGC[1]
        sim_train_0 = copy.deepcopy(sim_LLGC[0])
        sim_train_0[train_group]=-np.inf
        return sim_train_0, sim_LLGC[2], train_group



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

               

k=3
random.seed(554)
prec100_sl = np.zeros(k)
accuracy_sl = np.zeros(k)
NDCG_sl = np.zeros(k)
for i in range(k):
        t1 = time.perf_counter()
        results = get_sim_results(edges, number_nodes, groups, i, 0.1, 8, 10)
        print('t1 = ', time.perf_counter() - t1)
        prec100_sl[i] = get_prec100(results[0],results[1])
        accuracy_sl[i] = get_accuracy(results[0],results[1])
        NDCG_sl[i] = get_NDCG(results[0],results[1])
        print(i)

print(prec100_sl)
print(accuracy_sl)
print(NDCG_sl)


k=10
random.seed(554)
accuracy_sl = np.zeros(k)
accuracy_PPR = np.zeros(k)
accuracy_exp = np.zeros(k)
accuracy_LLGC = np.zeros(k)
for i in range(k):
        t1 = time.perf_counter()
        results_sl = get_sim_results(edges, number_nodes, groups, i, 0.1, 8, 10)
        print('t1 = ', time.perf_counter() - t1)
        t1 = time.perf_counter()
        results_PPR = get_PPR_results(edges, number_nodes, groups, i, 0.05, 8, 10)
        print('t2 = ', time.perf_counter() - t1)
        t1 = time.perf_counter()
        results_exp = get_exp_results(edges, number_nodes, groups, i, 0.1, 8, 10)
        print('t3 = ', time.perf_counter() - t1)
        t1 = time.perf_counter()
        results_LLGC = get_LLGC_results(edges, number_nodes, groups, i, 0.1, 8, 10)
        print('t4 = ', time.perf_counter() - t1)
        accuracy_sl[i] = get_accuracy(results_sl[0],results_sl[1])
        accuracy_PPR[i] = get_accuracy(results_PPR[0],results_PPR[1])
        accuracy_exp[i] = get_accuracy(results_exp[0],results_exp[1])
        accuracy_LLGC[i] = get_accuracy(results_LLGC[0],results_LLGC[1])
        print(i)

print(accuracy_sl)
print(accuracy_PPR)
print(accuracy_exp)
print(accuracy_LLGC)


importlib.reload(benchmark_methods)
k=4
random.seed(554)
accuracy_PPR = np.zeros(k)
accuracy_exp = np.zeros(k)
accuracy_LLGC = np.zeros(k)
for i in range(k):
        t1 = time.perf_counter()
        results_PPR = get_PPR_results(edges, number_nodes, groups, i, 0.05, 8, 10)
        print('t2 = ', time.perf_counter() - t1)
        t1 = time.perf_counter()
        results_exp = get_exp_results(edges, number_nodes, groups, i, 0.1, 8, 10)
        print('t3 = ', time.perf_counter() - t1)
        t1 = time.perf_counter()
        results_LLGC = get_LLGC_results(edges, number_nodes, groups, i, 0.9, 8, 10)
        print('t4 = ', time.perf_counter() - t1)
        accuracy_PPR[i] = get_accuracy(results_PPR[0],results_PPR[1])
        accuracy_exp[i] = get_accuracy(results_exp[0],results_exp[1])
        accuracy_LLGC[i] = get_accuracy(results_LLGC[0],results_LLGC[1])

print(accuracy_PPR)
print(accuracy_exp)
print(accuracy_LLGC)


get_sim_results(edges, number_nodes, groups, 0, 0.1, 8, 10)