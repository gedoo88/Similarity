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



def F_matrix(k, A, B, y):
        """returns the F matrix to k-th degree from A_tilde, P and y"""
        F = np.empty((len(y),2*k+1))
        F[:,0] = y
        x1 = A.dot(y) 
        F[:,1] = x1
        x2 = B.dot(y)
        F[:,2] = x2
        for x in range(1,k):
                l=x
                x1 = A.dot(x1)
                F[:,2*l+1] = x1
                x2 = B.dot(x2)
                F[:,2*l+2] = x2
        return F




def get_similarity_score(edges_data, n_nodes, groups, group_number, coef_regu, k, n_split):
        """returns predictions, indices rom the training group, indices from the testing group, the residuals and the 
        coefficients of the F matrix"""
        AP = AP_from_edges(edges_data, n_nodes)
        A_tilde = AP[0]
        P = AP[1]
        group = get_group(groups, group_number)
        random.shuffle(group)

        group_split = np.array_split(group,n_split)

        target = np.zeros(n_nodes)
        for x in range(0,len(group_split[0])):
                target[group_split[0][x]]=1
        
        nsplit = len(group_split[0])
        seed_split = np.array_split(group_split[0],nsplit)
        F = np.empty((n_nodes,2*k+1))
        F = F_matrix(k, A_tilde, P, target)
        F_restricted = np.delete(F, group_split[0], axis = 0)
        
        hide_seed = copy.deepcopy(target)
        hide_seed[seed_split[0]] = 0
        F_training = np.empty((nsplit,2*k+1))
        for x in range(0,nsplit):
                hide_seed = copy.deepcopy(target)
                hide_seed[seed_split[x]]=0
                F_training[x,:] = F_matrix(k,A_tilde,P,hide_seed)[seed_split[x],:]

        Lambda = coef_regu
        M = Lambda*np.identity(2*k+1) + np.transpose(F_training).dot(F_training) + np.transpose(F_restricted).dot(F_restricted)
        b = np.transpose(F_training).dot(np.ones(len(group_split[0])))
        c = np.linalg.solve(M,b)
        y_hat = F.dot(c)
        residu = (target-y_hat).dot(target-y_hat)
        test_group = np.delete(group,np.arange(len(group_split[0])))
        return y_hat, group_split[0], test_group, residu, c

  #donc travaille, ferme fb estpèce de petit écolier
  #TRAVAILLE 
  #Bon courage mon chéri
  #Bisous <3










