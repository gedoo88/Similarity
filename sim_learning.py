import math
import numpy as np
import pandas as pd
import os
import scipy
import scipy.sparse
import copy
import random
import time
from time import perf_counter



def AP_from_A(A, n_nodes):
        """returns the markovian transition and symetrically normalised adjacency matrix from the edges data and number of nodes""" 
        degrees = scipy.sparse.csr_matrix.sum(A, axis = 0)
        degrees_inv = 1/degrees
        
        D_inv = scipy.sparse.spdiags(degrees_inv, 0, n_nodes, n_nodes, format = "csr")
        D_inv_sqrt = scipy.sparse.spdiags(np.sqrt(degrees_inv), 0, n_nodes, n_nodes, format = "csr")

        A_tilde = D_inv_sqrt.dot(A).dot(D_inv_sqrt)
        P_transpose = A.dot(D_inv)

        return A_tilde, P_transpose 



def get_group_indices(groups_list,group_number):
        """from group edges, returns an array containing the indices corresponding to the group number selected"""
        file = open(groups_list, 'r')
        lines=file.readlines()
        group_indices = np.fromstring(lines[group_number],sep="\t", dtype = int)

        return group_indices



def F_matrix(k, A, B, y):
        """returns the F matrix to k-th degree from A_tilde, P and y"""
        F = np.empty((len(y),2*k+1))
        F[:,0] = y
        x1 = A.dot(y) 
        F[:,1] = x1
        x2 = B.dot(y)
        F[:,2] = x2
        for x in range(1,k):
                x1 = A.dot(x1)
                F[:,2*x+1] = x1
                x2 = B.dot(x2)
                F[:,2*x+2] = x2

        return F




def get_sim_score(A, n_nodes, groups, group_number, coef_regu, coef_pen, k, n_split):
        """returns predictions, indices rom the training group, indices from the testing group, the residuals and the 
        coefficients of the F matrix"""
        AP = AP_from_A(A, n_nodes)
        A_tilde = AP[0]
        P = AP[1]
        group = get_group_indices(groups, group_number)
        random.shuffle(group)
        group_split = np.array_split(group,n_split)
        train_group = group_split[0]
        test_group = np.delete(group,np.arange(len(train_group)))

        target = np.zeros(n_nodes)
        target[train_group]=1
        
        nsplit = 5
        seed_split = np.array_split(train_group,nsplit)
        F = np.empty((n_nodes,2*k+1))
        F = F_matrix(k, A_tilde, P, target)
        F_restricted = np.delete(F, train_group, axis = 0)

        F_training = np.empty((len(train_group),2*k+1))
        count = 0
        for x in range(nsplit):
                hide_seed = copy.deepcopy(target)
                hide_seed[seed_split[x]]=0
                hide_F = F_matrix(k,A_tilde,P,hide_seed)
                for i in range(len(seed_split[x])):
                        F_training[count+i,:] = hide_F[seed_split[x][i],:]
                count = count + len(seed_split[x])

        Lambda = coef_regu
        M = Lambda*np.identity(2*k+1) + np.transpose(F_training).dot(F_training) + coef_pen*np.transpose(F_restricted).dot(F_restricted)
        b = np.transpose(F_training).dot(np.ones(len(train_group))) 
        c = np.linalg.solve(M,b)
        y_hat = F.dot(c)
        y_hat[train_group]=-np.inf

        return y_hat, test_group, train_group, c











