import math
import numpy as np
import pandas as pd
import os
import scipy
import scipy.sparse
import copy
import random
import importlib
import sim_learning
importlib.reload(sim_learning)


edges = np.genfromtxt(r'C:\Users\Tim2\My Documents\GitHub\Similarity\edges.csv', delimiter=',', dtype=int)
nodes = np.genfromtxt(r'C:\Users\Tim2\My Documents\GitHub\Similarity\nodes.csv', delimiter=',', dtype=int)
groups = np.genfromtxt(r'C:\Users\Tim2\My Documents\GitHub\Similarity\group-edges.csv', delimiter=',', dtype=int)



def get_sim_results(edges_data, n_nodes, groups, group_number, k, coef_regu, n_split):
        S = sim_learning.get_similarity_score(edges_data, n_nodes, groups, group_number, k, coef_regu, n_split)
        train_group = S[1]
        sim_train_0 = copy.deepcopy(result[0])
        sim_train_0[train_group-1]=0
        return sim_train_0, S[2], train_group



def get_prec100(results, test_group):
        in_group= np.empty(shape = 0, dtype = bool)
        for i in np.argsort(-results)[:100]+1:
                in_group = np.append(in_group, i in test_group)
        return np.mean(in_group)



def get_accuracy(results, test_group):
        in_group= np.empty(shape = 0, dtype = bool)
        for i in np.argsort(-results)[:len(test_group)]+1:
                in_group = np.append(in_group, i in test_group)
        accuracy = np.mean(in_group)
        return np.mean(in_group)



def get_NDCG(results, test_group):
        IDCG = 0
        for i in range(len(test_group)):
                IDCG = IDCG + 1/(math.log(i+2))
        sum_NDCG = 0
        for i in range(len(nodes)-len(result[1])):
                if np.argsort(-results)[i] in test_group:
                        sum_NDCG = sum_NDCG + 1/(math.log(i+2))
        NDCG = sum_NDCG/IDCG 
        return NDCG



def get_groups_size(groups,number_groups):
        groups_size = np.zeros(number_groups)
        for i in range(len(groups)):
                groups_size[groups[i][1]-1] += 1
        return groups_size
                

groups_size = get_groups_size(groups,groups[-1][1])
groups_proportion = groups_size/len(nodes)


k=39
random.seed(554)
prec100 = np.zeros(k)
accuracy = np.zeros(k)
NDCG = np.zeros(k)
for i in range(k):
        results = get_sim_results(edges, len(nodes), groups, i+1, 8, 0.1, 10)
        prec100[i] = get_prec100(results[0],results[1])
        accuracy[i] = get_accuracy(results[0],results[1])
        NDCG[i] = get_NDCG(results[0],results[1])
        print(i)

print(prec100)
print(accuracy)
print(NDCG)


