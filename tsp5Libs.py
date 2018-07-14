import pandas as pd
import math, random

from time import time

import numpy as np
from numpy import array
from scipy.spatial import cKDTree

tree_refresh_interval = 10000 #every how many iterations to refresh the tree
max_randomity = 10 #how many edges will the alg choose from with randomity at max
tree_batch_size = 100 #size of the read ahead buffer from the tree
num_iterations = 20 #number of ants to explore the graph

file_path = ''
colony1 = 1 # ant colony
colony2 = 2 # ant colony
fer1 = 1 #feromone prima colonia
fer2 = 0.5 #feromone seconda colonia
logfile = 1 #0 = no log, 1 = standard output, 2 = file

#utility functions
def log(flag, msg):
    if flag:
        print(msg) 

def release_feromone (graph, node, edge, colony, direction, weight, all_weight, alpha):
    f = 'f1' if colony == colony1 else 'f2'
    
    #feromone reward when an edge is prime
    if direction == "reward":
        if colony == colony1:
            ff = (alpha * fer1) / (weight * all_weight)  if read_edge_prop (graph, node, edge, 'f1') == 0 else read_edge_prop (graph, node, edge, 'f1')
            update_edge(graph, node, edge, f1=ff * 1.50)
        else:
            ff = (alpha * fer2) / (weight * all_weight) if read_edge_prop (graph, node, edge, 'f2') == 0 else read_edge_prop (graph, node, edge, 'f2')
            update_edge(graph, node, edge, f2 = ff * 1.50)
    
    #feromone reset when another edge for the same node is marked as prime
    elif direction == "reset":
        is_not_me = lambda x: x != edge and x[0:2]=='e_'
        for e in filter(is_not_me, list(graph[node].keys())):
            if colony == colony1:
                update_edge(graph, node, e, f1=0)                
            else:
                update_edge(graph, node, e, f2=0)                
    
    #feromone left by exploring ants
    elif direction == "increase":
        ff = alpha / (weight * all_weight)
        ex_f = read_edge_prop(graph, node, edge, f)
        if colony == colony1:
            update_edge(graph, node, edge, f1 = ex_f + (ff*fer1))
        else:
            update_edge(graph, node, edge, f2 = ex_f + (ff*fer2))

def query_edges(T, node_x, node_y, NN, vis, node, origin, n):
    #0 doesn't exist, 1 should be the node itself
    i = 0
    nn_list = []
    while nn_list == [] or i == 0:
        i = i + 1 
        batch_nodes = range(1+tree_batch_size * (i-1), min(1+tree_batch_size * i, len(NN)+1))
        ww, ii = T.query([node_x, node_y], k=batch_nodes,n_jobs=-1) #i-th closest
        #n_list = [NN[i]['id'] for i in ii]
        #nn_list = list(filter(lambda x: not vis[int(x)] and x not in (origin, node), n_list))

        nw_list =[(NN[ii[i]]['id'], ww[i]) for i in range(len(ii))]
        nn_list = list(filter(lambda x: not vis[int(x[0])] and x[0] not in (origin, node), nw_list))

    nn_idx = int(random.uniform(0,min(max_randomity, len(nn_list)-1)))
    nn = nn_list[nn_idx][0]
    w = nn_list[nn_idx][1]
    e = 'e_' + node + ':' + nn 
    return {'edge':e, 'nn':nn, 'w':w}
    
def refresh_tree(NN, x, y, vis):
    x = [x[i] for i in range(len(x)) if not vis[int(NN[i]['id'])]]
    y = [y[i] for i in range(len(y)) if not vis[int(NN[i]['id'])]]
    NN = [NN[i] for i in range(len(NN)) if not vis[int(NN[i]['id'])]] 
    T = cKDTree(np.c_[x, y])
    return NN, x, y, T

#CRUD
def add_node(client, node):
    client[node]={}

def add_edge(client, s_node, t_node, edge, lb, w, f1, f2, n):
    # lb = label
    # w, f1, f2 = weight, feromone colony 1, feromone colony 2
    # n = already visited
    client[s_node][edge] = {'lb': lb, 'nn':t_node, 'w':w, 'f1':f1, 'f2':f2, 'n':n}
    #log(logfile, "Created edge {0}->{1}".format(s_node, t_node))

def read_edge_prop(client, s_node, edge, prop):
    return client[s_node][edge][prop]

def read_edge(client, s_node, edge):
    return client[s_node][edge]

def get_edges(client, node):
    filter_edges = lambda x: x[0:2]=='e_'
    return list(filter(filter_edges, client[node].keys()))

def update_edge(client, s_node, edge, t_node='', lb='', w=-1, f1=-1, f2=-1, n=-1):
    # lb = label
    # w, f1, f2 = weight, feromone colony 1, feromone colony 2
    # n = already visited
    if lb != '':
        client[s_node][edge]['lb']=lb
    if w != -1:
        client[s_node][edge]['w']=w
    if f1 != -1:
        client[s_node][edge]['f1']=f1
    if f2 != -1:
        client[s_node][edge]['f2']=f2
    if n >= 0:
        client[s_node][edge]['n']=n

def delete_edge(client, node, edge):
    del client[node][edge]
