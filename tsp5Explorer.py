import math, random
import _pickle as pkl
import sys, traceback
import csv
from itertools import compress
from time import time

from tsp5Libs import add_node, add_edge, update_edge, read_edge_prop, release_feromone
from tsp5Libs import file_path, colony1, colony2, log, fer1, fer2

from tsp5Libs import tree_refresh_interval, max_randomity, tree_batch_size, num_iterations
from tsp5Libs import query_edges, refresh_tree

import numpy as np
from numpy import array
from scipy.spatial import cKDTree

#parameters
verbose = False
source_f = file_path + 'santa_cities_full.csv'
dump_graph_file = file_path + 'santa_graph_full.pkl'
dump_param_file = file_path + 'santa_param_full.pkl'

def baseline_perf(N): #calculates the weight of the baselline solution
    max_x = max(N, key=lambda r:r['x'])
    max_y = max(N, key=lambda r:r['y']) 
    dy = max_y['y'] / math.sqrt(len(N))
    dx = max_x['x'] / math.sqrt(len(N))
    #assuming towns to be distributed uniformally in N rectangles of size dx * dy 
    #let's make the pessimistic assumption that every pair of nodes are positioned at the largest distance possible, 
    # either vertically and orizontally, and that we navigate through them orizzontally first and vertically later
    return math.sqrt(dx**2 + dy**2) * len(N)
    
def e_ant(g, N, origin, colony, alpha):    
    vis = [False] * len(g) #list of visited nodes
    stk = [] #stack
    s = () #stack element

    #init with origin
    node = origin
    all_weight = 0
    n = 1 #number of nodes added

    NN = [{'id':N[i]['id'], 'x':N[i]['x'], 'y':N[i]['y']} for i in range(len(N))]
    x = [N[i]['x'] for i in range(len(N))]
    y = [N[i]['y'] for i in range(len(N))]
    T = cKDTree(np.c_[x, y])
    
    while n < len(g):  
        if (n % 1000) == 0:
            log(verbose, '# nodes added: {0}'.format(str(n)))
        if (n % tree_refresh_interval) == 0:
            #refresh tree
            log(verbose, 'Refreshing the tree...')
            NN, x, y, T = refresh_tree(NN, x, y, vis)
        n_edge = query_edges(T, N[int(node)]['x'], N[int(node)]['y'], NN, vis, node, origin, n)
        if n_edge['w'] == 0:
            print('df')
        stk.append((node, n_edge))
        vis[int(node)] = True
        all_weight += n_edge['w']
        node = n_edge['nn']
        n += 1 #number of nodes added
    
    #add last node weight
    n = int(node)
    o = int(origin)
    all_weight += math.sqrt((N[n]['x']-N[o]['x'])**2 + (N[n]['y']-N[o]['y'])**2)

    # pop stack and create edges with feromone
    while stk != []:
        s = stk.pop() 
        s_node = s[0]
        edge = s[1]['edge']
        t_node = s[1]['nn']
        w = s[1]['w'] 
        if edge not in list(g[s_node].keys()): 
            add_edge(g, s_node, t_node, edge, 'way', w, 0, 0, 0) 
        release_feromone(g, s_node, edge, colony, 'increase', w, all_weight, alpha)

    return all_weight

def compute_stats(g, j):
    min_edges = len(N)
    max_edges = 0
    tot_edges = 0
    for n in g:
        node_edges = 0
        node_edges = len(g[n])
        tot_edges += node_edges
        if node_edges > 0 and node_edges < min_edges:
            min_edges = node_edges
        elif node_edges > max_edges:
            max_edges = node_edges
    avg_edges = tot_edges/len(N)
    return {'num':j, 'min':min_edges, 'max':max_edges, 'avg':avg_edges}

#global variables
N = [] #list of nodes red from source file
g = {} #dictionary to hold the graph

#reads nodes (id, x, y) from source file
log(True, 'Reading nodes from source file...') 

with open(source_f) as f:
    rows = csv.DictReader(f)
    for r in rows:
        node = {'id':r['id'], 'x':int(r['x']), 'y':int(r['y'])}
        N.append(node)

try:
    #add nodes
    log(True, 'Adding nodes to graph...')
    tot_nodes = len(N)
    for i in range(tot_nodes):
        add_node(g, N[i]['id'])
    
    #set parameters
    origin = N[0]['id'] #set origin
    alpha = baseline_perf(N)**2 / len(N)
    #feromone will be distributed proportionally to the following formula:
    # baseline / tot_w (inverse proportion vs total lenght, weighted over the baseline)
    # baseline/N * tot_w/w (inverse proportion to the edge's weight, weighted over the average baseline edge's weight)
    #this is baseline **2 / N * w = alpha / w (where alpha = baseline**2 / N)
    log(True, "alpha = {0}".format(str(alpha)))

    #add edges
    log(True, 'Adding edges to graph...')
    stat = []

    for j in range(num_iterations):
        w = e_ant(g, N, origin, colony1, alpha)
        log(True, 'Ant #{0} from Colony 1 discovered a way with weight {1}'.format(str(j+1), str(w)))
        st = (compute_stats(g, j+1))
        stat.append(st)
        
        
        w = e_ant(g, N, origin, colony2, alpha)
        log(True, 'Ant #{0} from Colony 2 discovered a way with weight {1}'.format(str(j+2), str(w)))
        st = (compute_stats(g, j+2))
        stat.append(st)
    
    log (True, "Merging feromone...")
    for n in g:
        for e in g[n]:
            ff1 = read_edge_prop(g,n,e,'f1')
            ff2 = read_edge_prop(g,n,e,'f2')
            update_edge (g, n, e, f1 = ff1 + 2*ff2, f2 = ff2 + ff1 / 2)

    log(True, 'Printing statistics (if collected) about #edges per node...')
    for s in stat:
        log(True, 'Step {0} - min {1} - max {2} - avg {3}'.format(s['num'], s['min'], s['max'], s['avg']))

    #serialize graph and parameters 
    log (True, 'Serializing the graph and parameters to disk...') 

    pkl.dump(g, open(dump_graph_file, 'wb')) 
    pkl.dump({'alpha' : alpha, 'origin' : origin, 'N' : N, 'Stat' : stat}, open(dump_param_file, 'wb')) 

except Exception as exc:
    print('There was an exception: {0}'.format(exc))
    traceback.print_exc(file=sys.stdout)
    sys.exit(1)