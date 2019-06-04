import math
import numpy as np
import pandas as pd
import os
import scipy
import scipy.sparse
import scipy.sparse.linalg
import copy
import random



def AP_from_edges(data_edges, n_nodes):
        """returns the markovian transition and symetrically normalised adjacency matrix from the edges data and number of nodes""" 
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
        """from group edges, returns an array containing the indices corresponding to the group number selected"""
        group_indices = np.empty(shape = 0, dtype = int)
        for x in range(0,len(groups_list)):
                if groups_list[x,1] == group_number:
                        group_indices = np.append(group_indices,groups_list[x,0])
        return group_indices



def get_PPR_prediction(edges_data, n_nodes, groups, group_number, alpha, n_split):

    AP = AP_from_edges(edges_data, n_nodes)
    P = AP[1]
    group = get_group(groups, group_number)
    random.shuffle(group)

    group_split = np.array_split(group,n_split)

    target = np.zeros(n_nodes)
    for x in range(0,len(group_split[0])):
        target[group_split[0][x]-1]=1
    
    S_inv = scipy.sparse.identity(n_nodes, format = "csr") - (1-alpha)*P
    y_hat = scipy.sparse.linalg.spsolve(S_inv, target) 
    test_group = np.delete(group,np.arange(len(group_split[0])))
    return y_hat, group_split[0], test_group



def get_exp_prediction(edges_data, n_nodes, groups, group_number, alpha, taylor_order, n_split):

    AP = AP_from_edges(edges_data, n_nodes)
    A_tilde = AP[0]
    group = get_group(groups, group_number)
    random.shuffle(group)

    group_split = np.array_split(group,n_split)

    target = np.zeros(n_nodes)
    for x in range(0,len(group_split[0])):
        target[group_split[0][x]-1]=1
    
    y_hat = target
    Delta = scipy.sparse.identity(n_nodes, format = "csr") - A_tilde
    for i in range(taylor_order):
        taylor_k = (-alpha*Delta).dot(target)
        y_hat = y_hat + taylor_k
        
    test_group = np.delete(group,np.arange(len(group_split[0])))
    return y_hat, group_split[0], test_group


def get_LLGC_prediction(edges_data, n_nodes, groups, group_number, alpha, n_split):

    AP = AP_from_edges(edges_data, n_nodes)
    A_tilde = AP[0]
    group = get_group(groups, group_number)
    random.shuffle(group)

    group_split = np.array_split(group,n_split)

    target = np.zeros(n_nodes)
    for x in range(0,len(group_split[0])):
        target[group_split[0][x]-1]=1
    
    S_inv = scipy.sparse.identity(n_nodes, format = "csr") - (1-alpha)*A_tilde
    y_hat = scipy.sparse.linalg.spsolve(S_inv, target) 
    test_group = np.delete(group,np.arange(len(group_split[0])))
    return y_hat, group_split[0], test_group

 
