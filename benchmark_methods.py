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

        row_ind[:nedges] = data_edges[:,0]
        row_ind[nedges:] = data_edges[:,1]
        col_ind[:nedges] = data_edges[:,1]
        col_ind[nedges:] = data_edges[:,0]

        Adj_coo = scipy.sparse.coo_matrix((np.ones(2*nedges, dtype = int),(row_ind,col_ind)), shape=(n_nodes,n_nodes))
        Adj_csr = scipy.sparse.coo_matrix.tocsr(Adj_coo)

        d = scipy.sparse.csr_matrix.sum(Adj_csr, axis = 0)
        D = scipy.sparse.spdiags(d, 0, n_nodes, n_nodes, format = "csr")
        d_inv = 1/d

        D_inv = scipy.sparse.spdiags(d_inv, 0, n_nodes, n_nodes, format = "csr")
        D_sqrt = scipy.sparse.spdiags(np.sqrt(d_inv), 0, n_nodes, n_nodes, format = "csr")

        A_tilde = D_sqrt.dot(Adj_csr).dot(D_sqrt)
        P = Adj_csr.dot(D_inv)
        return A_tilde, P 



def get_group(groups_list,group_number):
        """from group edges, returns an array containing the indices corresponding to the group number selected"""
        file = open(groups_list, 'r')
        lines=file.readlines()
        group_indices = np.fromstring(lines[group_number],sep="\t", dtype = int)
        return group_indices



def get_PPR_prediction(edges_data, n_nodes, groups, group_number, alpha, taylor_order, n_split):

    AP = AP_from_edges(edges_data, n_nodes)
    P = AP[1]
    group = get_group(groups, group_number)
    random.shuffle(group)

    group_split = np.array_split(group,n_split)

    target = np.zeros(n_nodes)
    for x in range(0,len(group_split[0])):
        target[group_split[0][x]]=1
    
    y_hat = copy.deepcopy(target)
    taylor_k = copy.deepcopy(target)
    for i in range(taylor_order):
        taylor_k = ((1-alpha)*P).dot(taylor_k)
        y_hat = y_hat + taylor_k
        
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
        target[group_split[0][x]]=1
    
    y_hat = copy.deepcopy(target)
    taylor_k = copy.deepcopy(target)
    Delta = scipy.sparse.identity(n_nodes, format = "csr") - A_tilde
    for i in range(taylor_order):
        taylor_k = (-(alpha/(i+1))*Delta).dot(taylor_k)
        y_hat = y_hat + taylor_k
        
    test_group = np.delete(group,np.arange(len(group_split[0])))
    return y_hat, group_split[0], test_group



def get_LLGC_prediction(edges_data, n_nodes, groups, group_number, alpha, taylor_order, n_split):

    AP = AP_from_edges(edges_data, n_nodes)
    A_tilde = AP[0]
    group = get_group(groups, group_number)
    random.shuffle(group)

    group_split = np.array_split(group, n_split)

    target = np.zeros(n_nodes)
    for x in range(0,len(group_split[0])):
        target[group_split[0][x]]=1
    
    y_hat = copy.deepcopy(target)
    taylor_k = copy.deepcopy(target)
    for i in range(taylor_order):
        taylor_k = (alpha*A_tilde).dot(taylor_k)
        y_hat = y_hat + taylor_k
        
    test_group = np.delete(group,np.arange(len(group_split[0])))
    return y_hat, group_split[0], test_group


