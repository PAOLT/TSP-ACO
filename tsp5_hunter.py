from tsp5Libs import file_path, colony1, colony2, log, fer1, fer2
from tsp5Libs import read_edge_prop, update_edge, release_feromone, read_edge, get_edges
import _pickle as pkl
import sys, traceback
import math, random

from tsp5Libs import query_edges, tree_refresh_interval, refresh_tree, add_edge#, query_edges_batch
import numpy as np
from numpy import array
from scipy.spatial import cKDTree

#parameters
verbose = False
dump_file_numb = ''
dump_graph_file_name = file_path + 'santa_graph_full' + dump_file_numb + '.pkl'
dump_param_file_name = file_path + 'santa_param_full.pkl'
final_graph_file_name = file_path + 'santa_graph_final_full' + dump_file_numb + '.pkl'

reduce_feromone_freq = 10 #every how many runs feromone is reduced
feromone_reduction = 0.05 #percentage of feromone reduction
dim_batch = 6

def scan_edges(graph, vis, node, origin, colony, batch_num, dim_batch, tot_feromone):
    if 'clu1' in list(graph[node].keys()) and colony == colony1:
        node_edges = graph[node]['clu1']['en']
        clu_w = graph[node]['clu1']['w']
        clu_n = graph[node]['clu1']['n']
    elif 'clu2' in list(graph[node].keys()) and colony == colony2:
        node_edges = graph[node]['clu2']['en']
        clu_w = graph[node]['clu2']['w']
        clu_n = graph[node]['clu2']['n']
    else:
        node_edges = node
        clu_w = 0 
        clu_n = 1
    
    clu = {'en':node_edges, 'n':clu_n, 'w':clu_w}
    
    if tot_feromone == 0:
        tot_f1, tot_f2 = 0, 0
        for e in get_edges(graph, node_edges):
            edg = read_edge(graph, node_edges, e)
            lb = edg['lb'] 
            nn = edg['nn'] 
            
            if nn != origin and ((colony == colony1 and lb not in ('blocked', "way2")) or (colony == colony2 and lb not in ('blocked', 'way1'))):
                    tot_f1 += edg['f1'] 
                    tot_f2 += edg['f2'] 
    else:
        tot_f1, tot_f2 = tot_feromone, tot_feromone
        #tot_f2 = tot_feromone 

    edges = []
    edge = {}
    f1, f2 = 0, 0
    tot_f = tot_f1 if colony == colony1 else tot_f2

    #for e in list(graph[node_edges].keys()):  
    for e in get_edges(graph, node_edges):  
        edg = read_edge(graph, node_edges, e)
        lb = edg['lb'] 
        nn = edg['nn']
        if not vis[int(nn)] and lb != 'blocked' and nn != origin and ((colony == colony1 and lb != "way2") or (colony == colony2 and lb != "way1")):
            #nn = edg['nn'] 
            w = edg['w'] 
            f1 = edg['f1'] 
            f2 = edg['f2'] 
            score = 0
            
            f = f1 if colony == colony1 else f2
            vs_f = f2 if colony == colony1 else f1  
                    
            if f > 3*vs_f and f > 0.4*tot_f:
                score = random.uniform(16,20)
            elif (f <= 3*vs_f and f > 0.4 * tot_f):
                score = random.uniform(11,15)
            elif (f > 3*vs_f and f <= 0.4*tot_f):
                score = random.uniform(6,10)    
            elif f > 0.8*vs_f:
                score = random.uniform(1,5)                    
            else:
                score = 0
                
            edge = {'edge':e, 'nn':nn, 'w':w, 'score':score}
            edges.append(edge) 

    if batch_num * dim_batch > len(edges): #estremo sx > fine lista edges 
        return clu, tot_f, []
    else:
        edges.sort(key = lambda t: t['score'], reverse = True)
        return clu, tot_f, edges[batch_num*dim_batch:(batch_num+1)*dim_batch]

def h_ant (g, N, origin, colony, alpha):
    stk = [] 
    clu = 'clu1' if colony == colony1 else 'clu2' 
    way = 'way1' if colony == colony1 else 'way2'
    
    node = origin
    n = 0
    all_weight = 0

    NN = [{'id':N[i]['id'], 'x':N[i]['x'], 'y':N[i]['y']} for i in range(len(N))]
    x = [N[i]['x'] for i in range(len(N))]
    y = [N[i]['y'] for i in range(len(N))]
    T = cKDTree(np.c_[x, y])

    forced = False

    while n < len(g):
        # if (n % tree_refresh_interval) == 0:
        #     log(verbose, 'Refreshing the tree...')
        #     NN, x, y, T = refresh_tree(NN, x, y, vis)

        if not vis[int(node)]:
            #scan edges - returns clu_n, clu_w, tot_fer, edges
            # edges = [{edge}]; edge = {'e':e, 'nn':nn, 'w':w, 'score':sc)
            vis[int(node)] = True
            cluster_config, tot_feromone, edges = scan_edges(g, vis, node, origin, colony, 0, dim_batch, 0) 
            if n % 1000 == 0:
                log(verbose, '{0} nodes visited. Now evaluating cluster {1} with {2} nodes'.format(str(n),node, cluster_config['n']))
            n = n + cluster_config['n']
            all_weight += (cluster_config['w'])

            if forced:
                forced = False
                edge = query_edges(T, N[int(node)]['x'], N[int(node)]['y'], NN, vis, node, origin, n) #return {'edge':e, 'nn':nn, 'w':ww[nn_idx]}
                add_edge(g, node, edge['nn'], edge['edge'], "way", edge['w'], 0, 0, 0)
                edges = [edge]

            if edges != []: 
                all_weight += edges[0]['w']
                stk.append({'cn':node, 'edges':edges, 'clu':cluster_config, 'batch_num':0, 'i':0, 't_f':tot_feromone})
                node = edges[0]['nn']
            elif n < len(g): #and not force
                vis[int(node)] = False
                n -= cluster_config['n']
                all_weight -= (cluster_config['w'])
                s = stk.pop()
                node = s['cn']
                #log(verbose, 'Stepping back to cluster {0}'.format(node))
            else: #and not force
                all_weight += math.sqrt((N[int(node)]['x']-N[int(origin)]['x'])**2 + (N[int(node)]['y']-N[int(origin)]['y'])**2)
                cluster_config = {'en':node, 'w':0, 'n':1} if clu not in g[node].keys() else g[node][clu]
                one_cluster = True
                vis[int(node)] = False
                log(verbose, '{0} nodes visited. Now closing the loop from node {1}'.format(str(n), node))
                log(verbose, '============================')
        else:
        #the node has been already visited once, so we are getting back from a node being an edge's target on its edges list
            #loading the step status from the step
            cluster_config = s['clu']
            edges = s['edges']
            batch_num = s['batch_num']
            i = s['i']
            tot_feromone = s['t_f']

            #releasing the weight of the failing edge and try the next edge
            all_weight = all_weight - edges[i]['w'] 
            #log (verbose, 'Back to node {0}'.format(node))            
            i = i+1 

            #in case i is out of range re: edges, let's load the next batch from the graph
            if i > len(edges)-1:
                batch_num += 1
                i = 0
                cluster_config, tot_feromone, edges = scan_edges(g, vis, node, origin, colony, batch_num, dim_batch, tot_feromone)
                    
            # case when the next batch is empty - as all the edges have been tested unsuccesfully, we have to force an edge to a node
            #might also be the case when we forced an edge to a node and its list of edges is empty - another edge has to be forced
            if edges == []:
                    #force the best edge 
                    if s['batch_num'] == 0:
                        edges = s['edges']
                    else:
                        batch_num = 0
                        cluster_config, tot_feromone, edges = scan_edges(g, vis, node, origin, colony, batch_num, dim_batch, tot_feromone)
                    i = 0
                    forced = True
            
            all_weight += edges[i]['w'] 
            stk.append({'cn':node, 'edges':edges, 'clu': cluster_config,'batch_num':batch_num, 'i':i, 't_f':tot_feromone})
            node = edges[i]['nn'] 
            
            #OLD CODE with step back
            # if edges == []:
            #         vis[int(node)] = False
            #         n -= cluster_config['n']
            #         all_weight -= cluster_config['w']
            #         log (verbose, 'Releasing node {0}'.format(node))
            #         s = stk.pop()
            #         node = s['cn']
            # else:
            #     all_weight += edges[i]['w'] 
            #     stk.append({'cn':node, 'edges':edges, 'clu': cluster_config,'batch_num':batch_num, 'i':i, 't_f':tot_feromone})
            #     node = edges[i]['nn']         

    while stk != []:
        s = stk.pop()  
        #stk.append({'cn':n, 'edges':es, 'clu':clu, 'batch_num':bn, 'i':i, 't_f':tf})
        #edges = [{'edge':e, 'nn':nn, 'w':w, 'score':score}]
        node = s['cn']
        node_config = s['clu']
        edge = s['edges'][s['i']]['edge']
        nn = read_edge_prop(g, node_config['en'], edge, 'nn')          
        
        tot_feromone = s['t_f']
        
        #release feromone
        release_feromone (g, node_config['en'], edge, colony, 'increase', read_edge_prop(g, node_config['en'], edge, 'w'), all_weight, alpha)
        
        #clustering
        vis[int(node)] = False 
        edg = read_edge(g, node_config['en'], edge)
        f = edg['f1'] if colony == colony1 else edg['f2']
        vs_f = edg['f2'] if colony == colony1 else edg['f1']

        is_prime = (f > 3*vs_f and f > 0.4*tot_feromone)

        if is_prime:
            en = cluster_config['en']
            w = node_config['w'] + cluster_config['w'] + edg['w']
            #w = node_config['w'] + cluster_config['w'] + read_edge_prop(g, node_config['en'], edge, 'w')
            n = node_config['n'] + cluster_config['n']

            if clu in g[node].keys():
                del g[node][clu]
            
            if clu in g[nn].keys():
                del g[nn][clu]

            cluster_config =  {'en':en, 'w': w, 'n': n}
            vis[int(nn)] = True
            update_edge(g, node_config['en'], edge, lb=way)
            release_feromone(g, node_config['en'], edge, colony, 'reset', 0, 0, 0)
            
            # filter_edges = lambda x: x[0:2]=='e_'
            # for e in filter(filter_edges, g[nn].keys()):
            for e in get_edges(g, nn):
                if read_edge_prop(g, nn, e, 'nn') == node_config['en']:
                    update_edge(g, nn, e, lb="blocked")
                    break  
        
        else:
            if cluster_config['en'] != nn:
                g[nn][clu] = cluster_config 
                log(True, 'CLUSTER! It has {0} nodes'.format(str(cluster_config['n'])))
            #cluster_config = {'en':node, 'w':read_edge_prop(g, node, edge, 'w'), 'n':1} if clu not in g[node].keys() else g[node][clu]
            cluster_config = node_config
            

        #termination condition
        one_cluster = one_cluster if is_prime else False

    else:
        if cluster_config['en'] != origin:
            g[origin][clu] = cluster_config

    return one_cluster, all_weight


#deserialize graph and params
log(True, 'Loading data...')
file_handle = open(dump_graph_file_name, 'rb')
g = pkl.load(file_handle)
file_handle.close()

file_handle = open(dump_param_file_name, 'rb')
prm = pkl.load(file_handle)
file_handle.close()

alpha = prm['alpha']
origin = prm['origin']
N = prm['N']

#run hunters
try:
    term = False
    i = 0
    vis = [False] * len(g)  
    log(True, 'Starting ants...') 
    while not term:
        term, weight = h_ant(g, N, origin, colony1, alpha)
        print('Ant #{0} in colony #1 returned with weight {1}'.format(str(i), str(weight)))
        #reduce feromone 
        # if (i % reduce_feromone_freq) == 0:
        #     for n in g:
        #         for e in g[n]:
        #             new_f1 = read_edge_prop(g, n, e, 'f1') * (1-feromone_reduction)
        #             update_edge(g, n, e, f1 = new_f1)
        i +=1 
    print("End of colony #1 with weight {0}".format(str(weight)))

    term = False
    i = 0
    vis = [False] * len(g)   
    while not term:
        term, weight = h_ant(g, N, origin, colony2, alpha)
        print('Ant #{0} in colony #2 returned with weight {1}'.format(str(i), str(weight)))
        #reduce feromone 
        if i % reduce_feromone_freq == 0:
            for n in g:
                for e in g[n]:
                    new_f2 = read_edge_prop(g, n, e, 'f2')
                    update_edge(g, n, e, f2 = new_f2) 
        i +=1 
    print("End of colony #2 with weight {0}".format(str(weight)))

    #verify the solution

    #serialize graph and parameters 
    log (True, 'Serializing the graph to disk...') 

    pkl.dump(g, open(final_graph_file_name, 'wb')) 
    pkl.dump(g, open(final_graph_file_name, 'wb')) 
        
except Exception as exc:
    print('There was an exception: {0}'.format(exc))
    traceback.print_exc(file=sys.stdout)
    sys.exit(1)