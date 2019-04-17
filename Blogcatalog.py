import numpy as np
import pandas as pd
import os
import scipy
import scipy.sparse
import copy



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

A_tilde = D_sqrt.dot(Adj_csr).dot(D_sqrt)
P = D_inv.dot(Adj_csr)

group_1 = np.empty(shape = 0, dtype = int)

for x in range(0,len(groups)):
    if groups[x,1] == 1:
        group_1 = np.append(group_1,groups[x,0])

seed = np.array(group_1[0:len(group_1)//10])
y = np.zeros(size)

for x in range(0,len(seed)):
        y[seed[x]-1]=1




def F_matrix(k, A, B, y):
        F = np.column_stack((y,A.dot(y),B.dot(y)))
        for x in range(1,k):
                l=x
                F = np.column_stack((F, A.dot(F[:,2*l-1])))
                F = np.column_stack((F, B.dot(F[:,2*l])))
        return F

k = 8

nsplit = 5
seed_split = np.array_split(seed,nsplit)

F = F_matrix(k,A_tilde,P, y)

F_restricted = np.delete(F, seed, axis = 0)

hide_seed = copy.deepcopy(y)
hide_seed[seed_split[0]-1] = 0
F_training = F_matrix(k,A_tilde,P,hide_seed)[seed_split[0],:]
for x in range(1,nsplit):
        hide_seed = copy.deepcopy(y)
        hide_seed[seed_split[x]-1]=0
        F_training = np.row_stack((F_training,F_matrix(k,A_tilde,P,hide_seed)[seed_split[x],:]))




Lambda = 0.1

M = Lambda*np.identity(2*k+1) + np.transpose(F_training).dot(F_training) + np.transpose(F_restricted).dot(F_restricted)
b = np.transpose(F_training).dot(np.ones(len(seed)))

c = np.linalg.solve(M,b)

S = F.dot(c)

ind = np.argpartition(S, -20)[-20:]

