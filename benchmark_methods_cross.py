import math
import numpy as np
import pandas as pd
import os
import scipy
import scipy.sparse
import scipy.sparse.linalg
import copy
import random



def AP_from_A(A, n_nodes):
        """returns the markovian transition and symetrically normalised adjacency matrix from the edges data and number of nodes""" 
        degrees = scipy.sparse.csr_matrix.sum(A, axis = 0)
        degrees_inv = 1/degrees
        
        D_inv = scipy.sparse.spdiags(degrees_inv, 0, n_nodes, n_nodes, format = "csr")
        D_inv_sqrt = scipy.sparse.spdiags(np.sqrt(degrees_inv), 0, n_nodes, n_nodes, format = "csr")

        A_tilde = D_inv_sqrt.dot(A).dot(D_inv_sqrt)
        P_transpose = A.dot(D_inv)

        return A_tilde, P_transpose 



def get_group(groups_list,group_number):
        """from group edges, returns an array containing the indices corresponding to the group number selected"""
        file = open(groups_list, 'r')
        lines=file.readlines()
        group_indices = np.fromstring(lines[group_number],sep="\t", dtype = int)
        return group_indices



def get_PPR_score(A, n_nodes, groups, group_number, taylor_order, n_split):

    alpha = np.array([0.05,0.1,0.15,0.2])
    AP = AP_from_A(A, n_nodes)
    P = AP[1]
    group = get_group(groups, group_number)
    random.shuffle(group)

    group_split = np.array_split(group, n_split)

    #5-fold split for the cross validation
    split_cv = np.array_split(group_split[0], 5)

    target = np.zeros(n_nodes)
    target[group_split[0]]=1

    #Compute the NDCG for each hidden 5-fold for every parameter in range alpha and chose the best
    NDCG_mean = np.empty((len(alpha)))
    for i in range(len(alpha)):
        NDCG = np.empty(5)
        for j in range(5):
            y_cv = copy.deepcopy(target)
            y_cv[split_cv[j]] = 0
            taylor_k = copy.deepcopy(y_cv)
            for z in range(taylor_order):
                taylor_k = ((1-alpha[i])*P).dot(taylor_k)
                y_cv = y_cv + taylor_k
            y_cv[split_cv[(j+1)%5]]=-np.inf #set to -infinity so that known labels are at the end of the ranking
            y_cv[split_cv[(j+2)%5]]=-np.inf 
            y_cv[split_cv[(j+3)%5]]=-np.inf 
            y_cv[split_cv[(j+4)%5]]=-np.inf  
            x = np.arange(len(split_cv[j]))
            IDCG = np.sum(1/np.log(x+2))
            sum_NDCG = 0
            z1 = np.empty(len(split_cv[j]), dtype = int)
            z2 = np.argsort(-y_cv)
            for z in range(len(split_cv[j])):
                z1[z] = np.argwhere(z2 == split_cv[j][z])
            sum_NDCG = np.sum(1/np.log(z1+2))
            NDCG[j] = sum_NDCG/IDCG  
        NDCG_mean[i] = np.mean(NDCG)
    alpha_cv = alpha[np.argmax(NDCG_mean)]
    print(alpha_cv)
    
    y_hat = copy.deepcopy(target)
    taylor_k = copy.deepcopy(target)
    for i in range(taylor_order):
        taylor_k = ((1-alpha_cv)*P).dot(taylor_k)
        y_hat = y_hat + taylor_k
        
    test_group = np.delete(group,np.arange(len(group_split[0])))
    y_hat[group_split[0]]=-np.inf

    return y_hat, test_group, group_split[0]



def get_exp_score(A, n_nodes, groups, group_number, taylor_order, n_split):

    alpha = np.array([0.1,0.5,1,2,10])
    AP = AP_from_A(A, n_nodes)
    A_tilde = AP[0]
    Delta = scipy.sparse.identity(n_nodes, format = "csr") - A_tilde
    group = get_group(groups, group_number)
    random.shuffle(group)

    group_split = np.array_split(group, n_split)

    split_cv = np.array_split(group_split[0], 5)

    target = np.zeros(n_nodes)
    target[group_split[0]]=1

    NDCG_mean = np.empty((len(alpha)))
    for i in range(len(alpha)):
        NDCG = np.empty(5)
        for j in range(5):
            y_cv = copy.deepcopy(target)
            y_cv[split_cv[j]] = 0
            taylor_k = copy.deepcopy(y_cv)
            for z in range(taylor_order):
                taylor_k = (-(alpha[i]/(z+1))*Delta).dot(taylor_k)
                y_cv = y_cv + taylor_k
            y_cv[split_cv[(j+1)%5]]=-np.inf 
            y_cv[split_cv[(j+2)%5]]=-np.inf 
            y_cv[split_cv[(j+3)%5]]=-np.inf 
            y_cv[split_cv[(j+4)%5]]=-np.inf  
            x = np.arange(len(split_cv[j]))
            IDCG = np.sum(1/np.log(x+2))
            sum_NDCG = 0
            z1 = np.empty(len(split_cv[j]), dtype = int)
            z2 = np.argsort(-y_cv)
            for z in range(len(split_cv[j])):
                z1[z] = np.argwhere(z2 == split_cv[j][z])
            sum_NDCG = np.sum(1/np.log(z1+2))
            NDCG[j] = sum_NDCG/IDCG  
        NDCG_mean[i] = np.mean(NDCG)
    alpha_cv = alpha[np.argmax(NDCG_mean)]
    print(alpha_cv)
    
    y_hat = copy.deepcopy(target)
    taylor_k = copy.deepcopy(target)
    for i in range(taylor_order):
        taylor_k = (-(alpha_cv/(i+1))*Delta).dot(taylor_k)
        y_hat = y_hat + taylor_k
        
    test_group = np.delete(group,np.arange(len(group_split[0])))
    y_hat[group_split[0]]=-np.inf

    return y_hat, test_group, group_split[0]



def get_LLGC_score(A, n_nodes, groups, group_number, taylor_order, n_split):

    alpha = np.array([0.1,0.3,0.5,0.7,0.9])
    AP = AP_from_A(A, n_nodes)
    A_tilde = AP[0]
    group = get_group(groups, group_number)
    random.shuffle(group)

    group_split = np.array_split(group, n_split)

    split_cv = np.array_split(group_split[0], 5)

    target = np.zeros(n_nodes)
    target[group_split[0]]=1

    NDCG_mean = np.empty((len(alpha)))
    for i in range(len(alpha)):
        NDCG = np.empty(5)
        for j in range(5):
            y_cv = copy.deepcopy(target)
            y_cv[split_cv[j]] = 0
            taylor_k = copy.deepcopy(y_cv)
            for z in range(taylor_order):
                taylor_k = (alpha[i]*A_tilde).dot(taylor_k)
                y_cv = y_cv + taylor_k
            y_cv[split_cv[(j+1)%5]]=-np.inf 
            y_cv[split_cv[(j+2)%5]]=-np.inf 
            y_cv[split_cv[(j+3)%5]]=-np.inf 
            y_cv[split_cv[(j+4)%5]]=-np.inf  
            x = np.arange(len(split_cv[j]))
            IDCG = np.sum(1/np.log(x+2))
            sum_NDCG = 0
            z1 = np.empty(len(split_cv[j]), dtype = int)
            z2 = np.argsort(-y_cv)
            for z in range(len(split_cv[j])):
                z1[z] = np.argwhere(z2 == split_cv[j][z])
            sum_NDCG = np.sum(1/np.log(z1+2))
            NDCG[j] = sum_NDCG/IDCG  
        NDCG_mean[i] = np.mean(NDCG)
    alpha_cv = alpha[np.argmax(NDCG_mean)]
    print(alpha_cv)
    
    y_hat = copy.deepcopy(target)
    taylor_k = copy.deepcopy(target)
    for i in range(taylor_order):
        taylor_k = (alpha_cv*A_tilde).dot(taylor_k)
        y_hat = y_hat + taylor_k
        
    test_group = np.delete(group,np.arange(len(group_split[0])))
    y_hat[group_split[0]]=-np.inf

    return y_hat, test_group, group_split[0]



