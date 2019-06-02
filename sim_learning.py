import numpy as np
import pandas as pd
import os
import scipy
import scipy.sparse
import copy
import random



def AP_from_edges(data_edges, n_nodes):
        nedges = len(data_edges)
        row_ind = np.zeros(2*nedges, dtype = int)
        col_ind = np.zeros(2*nedges, dtype = int)

        for x in range(0,nedges): 
                row_ind[x]=data_edges[x,0]-1
                row_ind[nedges+x]=data_edges[x,1]-1
                col_ind[x]=data_edges[x,1]-1
                col_ind[nedges+x]=data_edges[x,0]-1

        Adj_coo = scipy.sparse.coo_matrix((np.ones(2*nedges, dtype = int),(row_ind,col_ind)), shape=(n_nodes,n_nodes))
        Adj_csr = scipy.sparse.coo_matrix.tocsr(Adj_coo)

        d = scipy.sparse.csr_matrix.sum(Adj_csr, axis = 0)
        D = scipy.sparse.spdiags(d, 0, n_nodes, n_nodes, format = "csr")
        d_inv = 1/d

        D_inv = scipy.sparse.spdiags(d_inv, 0, n_nodes, n_nodes, format = "csr")
        D_sqrt = scipy.sparse.spdiags(np.sqrt(d_inv), 0, n_nodes, n_nodes, format = "csr")

        A_tilde = D_sqrt.dot(Adj_csr).dot(D_sqrt)
        P = D_inv.dot(Adj_csr)
        return A_tilde, P 



def get_group(groups_list,group_number):
        group_indices = np.empty(shape = 0, dtype = int)
        for x in range(0,len(groups_list)):
                if groups_list[x,1] == group_number:
                        group_indices = np.append(group_indices,groups_list[x,0])
        return group_indices



def F_matrix(k, A, B, y):
        F = np.column_stack((y,A.dot(y),B.dot(y)))
        for x in range(1,k):
                l=x
                F = np.column_stack((F, A.dot(F[:,2*l-1])))
                F = np.column_stack((F, B.dot(F[:,2*l])))
        return F




def get_similarity_score(edges_data, n_nodes, groups, group_number, k, coef_regu, n_split):
        AP = AP_from_edges(edges_data, n_nodes)
        A_tilde = AP[0]
        P = AP[1]
        group = get_group(groups, group_number)
        random.shuffle(group)

        group_split = np.array_split(group,n_split)

        target = np.zeros(n_nodes)
        for x in range(0,len(group_split[0])):
                target[group_split[0][x]-1]=1

        nsplit = len(group_split[0])
        seed_split = np.array_split(group_split[0],nsplit)

        F = F_matrix(k, A_tilde, P, target)
        F_restricted = np.delete(F, group_split[0]-1, axis = 0)

        hide_seed = copy.deepcopy(target)
        hide_seed[seed_split[0]-1] = 0
        F_training = F_matrix(k,A_tilde,P,hide_seed)[seed_split[0]-1,:]
        for x in range(1,nsplit):
                hide_seed = copy.deepcopy(target)
                hide_seed[seed_split[x]-1]=0
                F_training = np.row_stack((F_training,F_matrix(k,A_tilde,P,hide_seed)[seed_split[x],:]-1))

        Lambda = coef_regu
        M = Lambda*np.identity(2*k+1) + np.transpose(F_training).dot(F_training) + np.transpose(F_restricted).dot(F_restricted)
        b = np.transpose(F_training).dot(np.ones(len(group_split[0])))
        c = np.linalg.solve(M,b)
        S = F.dot(c)
        residu = (target-S).dot(target-S)
        test_group = np.delete(group,np.arange(len(group_split[0])))
        return S, group_split[0], test_group, residu, c

  






