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



groups = np.genfromtxt(r'C:\Users\Tim2\My Documents\GitHub\Similarity\group-edges.csv', delimiter=',', dtype=int)

def get_groups_size(groups,number_groups):
        groups_size = np.zeros(number_groups)
        for i in range(len(groups)):
                groups_size[groups[i][1]-1] += 1
        return groups_size

def get_group_formatbc(groups_list,group_number):
        """from group edges, returns an array containing the indices corresponding to the group number selected"""
        group_indices = np.empty(shape = 0, dtype = int)
        for x in range(len(groups_list)):
                if groups_list[x,1] - 1 == group_number:
                        group_indices = np.append(group_indices,groups_list[x,0])
        return group_indices   

groups_size = get_groups_size(groups,groups[-1][1])

large_groups = np.argwhere(groups_size >= 100)

with open(r'C:\Users\Tim2\My Documents\Python-proj\blogcat_contiguous_large_groups.txt', 'wb') as f:
        for i in range(len(large_groups)):
                group = get_group_formatbc(groups,i)-1
                print(i)
                np.savetxt(f, np.reshape(group,(1,-1)), fmt='%i')



edges = np.genfromtxt(r'C:\Users\Tim2\My Documents\GitHub\Similarity\edges.csv', delimiter=',', dtype=int)
edges = edges - 1
with open(r'C:\Users\Tim2\My Documents\Python-proj\blogcat_ungraph.txt', 'wb') as f:
        np.savetxt(f, edges, fmt='%i') 
                




file = open(r'C:\Users\Tim2\My Documents\GitHub\Similarity\com-dblp.ungraph.txt', 'r')
file.readline()
file.readline()
file.readline()
file.readline()
row = 0
for line in file:
        edges[row] = np.fromstring(line, sep="\t")
        row += 1
indices = np.unique(edges)

flat = np.ndarray.flatten(edges)

for i in range(len(indices)):
        flat[np.argwhere(flat == indices[i])] = i

newindices = np.reshape(flat,(-1,2))
np.savetxt('dblp_contiguous_ungraph', newindices, fmt='%i')




group_size = np.empty(13477, dtype = int)
row = 0
for line in open(r'C:\Users\Tim2\My Documents\GitHub\Similarity\com-dblp.all.cmty.txt'):
        group_indices = np.fromstring(line,sep="\t", dtype = int)
        group_size[row] = len(group_indices)
        row += 1

large_groups = np.argwhere(group_size >= 1000)

with open(r'C:\Users\Tim2\My Documents\Python-proj\dblp_contiguous_large_groups.txt', 'wb') as f:
        for i in range(149):
                group = np.fromstring(lines[np.int(large_groups[i])],sep="\t", dtype = int)
                for j in range(len(group)):
                        group[j] = np.argwhere(indices == group[j])
                group = np.reshape(group,(1,-1))
                print(i)
                np.savetxt(f, group, fmt='%i', delimiter="\t")

