import numpy as np
import pandas as pd
import os
import scipy
import scipy.sparse



edges = np.genfromtxt('edges.csv', delimiter=',', dtype=int)
nedges = len(edges)
size = 10312
groups = np.genfromtxt('group-edges.csv', delimiter=',', dtype=int)


row_ind = np.zeros(2*nedges, dtype = int)
col_ind = np.zeros(2*nedges, dtype = int)
data = np.zeros(2*nedges, dtype = int)

for x in range(0,nedges): 
    row_ind[x]=edges[x,0]-1
    row_ind[nedges+x]=edges[x,1]-1
    col_ind[x]=edges[x,1]-1
    col_ind[nedges+x]=edges[x,0]-1
    data[x]=1
    data[nedges+x]=1

Adj_coo = scipy.sparse.coo_matrix((data,(row_ind,col_ind)), shape=(size,size))
Adj_csr = scipy.sparse.coo_matrix.tocsr(Adj_coo)

d = scipy.sparse.csr_matrix.sum(Adj_csr, axis = 0)
D = scipy.sparse.spdiags(d, 0, size, size, format = "csr")
d_inv = 1/d

D_inv = scipy.sparse.spdiags(d_inv, 0, size, size, format = "csr")
D_sqrt = scipy.sparse.spdiags(np.sqrt(d_inv), 0, size, size, format = "csr")

A_tilde = D_sqrt*Adj_csr*D_sqrt
P = D_inv*Adj_csr

group_1 = np.empty(shape = 0, dtype = int)

for x in range(0,len(groups)):
    if groups[x,1] == 1:
        group_1 = np.append(group_1,groups[x,0])

seed = np.array(group_1[0:len(group_1)//10])
y = np.zeros()

k = 8

F = np.array()

