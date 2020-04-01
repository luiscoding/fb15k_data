from ast import literal_eval
import time
import networkx as nx
from tqdm import tqdm
import multiprocessing
import collections
import json
import os
import jsonlines
import sys


def all_simple_paths(G,source,target,cutoff):
    if source not in G:
        raise nx.NodeNotFound('source node %s not in graph' % source)
    if target in G:
        targets = {target}
    else:
        try:
            targets = set(target)
        except TypeError:
            raise nx.NodeNotFound('target node %s not in graph' % target)
    if source in targets:
        return []
    if cutoff is None:
        cutoff = len(G) - 1
    if cutoff < 1:
        return []
    if G.is_multigraph():
        return _all_simple_paths_multigraph(G, source, targets, cutoff)
    else:
        return None

def _all_simple_paths_multigraph(G, source, targets, cutoff):
    visited = collections.OrderedDict.fromkeys([(source,-1)])
    stack = [((v,c) for u, v, c in G.edges(source,keys = True))]

    while stack:
        children = stack[-1]
        child = next(children, None)

        if child is None:
            stack.pop()
            visited.popitem()
        elif len(visited) < cutoff:
            if child in visited:
                continue
            if child[0] in targets:
                yield list(visited) + [child]
            visited[child] = None
            if targets - set(visited.keys()):
                stack.append((v,c) for u, v, c in G.edges(child[0],keys = True))
            else:
                visited.popitem()
        else:  # len(visited) == cutoff:
            for target in targets - set(visited.keys()):
                count = ([child] + list(children)).count(target)
                for i in range(count):
                    yield list(visited) + [target]
            stack.pop()
            visited.popitem()

def stringify_keys(d):
    """Convert a dict's keys to strings if they are not."""
    for key in d.keys():

        # check inner dict
        if isinstance(d[key], dict):
            value = stringify_keys(d[key])
        else:
            value = d[key]

        # convert nonstring to string if needed
        if not isinstance(key, str):
            try:
                d[str(key)] = value
            except Exception:
                try:
                    d[repr(key)] = value
                except Exception:
                    raise
            # delete old key
            del d[key]
    return d

def read_rules(f):
    rules ={}
    with jsonlines.open(f) as reader:
        for obj in reader:
            for k in obj:
                rules[k] = obj[k]["rules"].keys()
    for k in rules:
        rule = []
        for ii in rules[k]:
            rule.append(literal_eval(ii.strip("()")))
        rules[k] = rule
    return rules

def check_path_valid(G,source,tail,rules):
    rule_set = set()
    rule_set_2  = set()
    rule_set_3  = set()
    for ii in rules:
        rule_set.add(ii[0])
        rule_set_2.add(ii[1])
        rule_set_3.add(ii[2])
    keep_edges = []
    paths = all_simple_paths(G,source,tail,cutoff =4)
    # print(len(list(paths)))
    for pp in paths:
        if(len(pp)==4 and pp[1][1] in rule_set and pp[2][1] in rule_set_2 and pp[3][1] in rule_set_3):
            r_path =[pp[1][1],pp[2][1],pp[3][1]]
            r_edges = [(pp[0][0],pp[1][0],pp[1][1]),(pp[1][0],pp[2][0],pp[2][1]),(pp[2][0],pp[3][0],pp[3][1])]
            # for i in range(1,len(pp)):
            #     r_path.append(pp[i][1])
            #     r_edges.append((pp[i-1][0],pp[i][0],pp[i][1]))
            if(r_path in rules):
                keep_edges.extend(r_edges)
    # print("this is the keep ",keep_edges)
    return keep_edges


def check_connected_subgraph2(G,rules,pair,path):
    head = pair[0]
    tail = pair[1]
    all_rules = set()
    for ii in rules:
        for jj in ii:
            all_rules.add(jj)

    if(G.has_node(head)):
        F_edges= nx.bfs_edges(G,head,depth_limit=3)
        F_nodes = list(set([head] + [v for u, v in F_edges] +[u for u, v in F_edges]))
        F =nx.MultiDiGraph(G.subgraph(F_nodes))
        remove_edges = set()
        fg1_name = path+"_".join(pair).replace("/","-")+"_fg_1.0.json"
        with open(fg1_name,"w") as f:
            json.dump(list(F.edges),f)
        for u,v,k in F.edges:
            if k not in all_rules:
                remove_edges.add((u,v,k))
        for ii in remove_edges:
            F.remove_edge(ii[0],ii[1],ii[2])

        keep_edges =[]
        for s in [head]:
            for t in (nx.bfs_tree(F, head)).nodes:
                if(nx.has_path(F,s,t)):
                    valid_edges= check_path_valid(F,s,t,rules)
                    keep_edges.extend(valid_edges)
                    # if len(valid_edges)>0:
                    #     print(keep_edges)

        # H = F.edge_subgraph(keep_edges)
        H = nx.MultiDiGraph()
        # G = nx.DiGraph()
        # G.add_nodes_from(all_nodes)
        H.add_edges_from(keep_edges)

        if(H.has_node(head) and H.has_node(tail) and nx.has_path(H,head,tail)):
            count = 1
            print(head,"contains",tail)
        else:
            count = 0
            print(head,"not contains",tail)
        edge_num = H.size()
        node_num = len(list(H.nodes()))
        stt = "_".join(pair).replace("/","-")
        f_name = path+stt+".json"
        print(f_name)
        with open(f_name,"w") as f:
            json.dump(keep_edges,f)
    if(len(keep_edges)==0):
        return 0, count,edge_num,node_num
    else:
        return 1,count,edge_num,node_num



def construct_original_graph():
    head_nodes = list()
    tail_nodes = list()
    edges = list()
    rel_dict = dict()
    path = "/home/luzhang/Desktop/luzhang/2019fall/Project"
    # read train data
    with open (path+"/kb_graph/fb15k_237/train.txt") as fin:
        for line in fin:
            line_list = line.strip('\n').split('\t')
            head_nodes.append(line_list[0])
            tail_nodes.append(line_list[2])
            edges.append((line_list[0],line_list[2],line_list[1]))

    all_nodes = set(head_nodes).union(set(tail_nodes))
    # networkx construct graph
    G = nx.MultiDiGraph()
    # G = nx.DiGraph()
    G.add_nodes_from(all_nodes)
    G.add_edges_from(edges)


    head_nodes_dev = list()
    tail_nodes_dev = list()
    edges_dev = list()
    rel_dict_dev = dict()
    rell=set()

    with open (path+"/kb_graph/fb15k_237/valid.txt") as fin:
        for line in fin:
            line_list = line.strip('\n').split('\t')
            head_nodes_dev.append(line_list[0])
            tail_nodes_dev.append(line_list[2])
            edges_dev.append((line_list[0],line_list[2],line_list[1]))
            rel_dict_dev[(line_list[0],line_list[2])] = line_list[1]
            rell.add(line_list[1])
    return G,edges,edges_dev


if __name__ == "__main__":
    d = int(1)
    G,edges,edges_dev = construct_original_graph()

    print(d)
    path ="./output_filter_graph_2.0/"+ time.strftime("%Y%m%d-%H%M%S")+ 'filter_graph_'+str(d)+'/'
    access_rights = 0o777

    try:
        os.makedirs(path, access_rights)
    except:
        print("create direct failed")

    file_name = "./19_09_29_10_2019/test_train.jsonl"
    rules = read_rules(file_name)['/tv/tv_program/languages']
    print(len(rules))

    import torch

    # with open("/home/luzhang/Desktop/luzhang/2019fall/Project/kb_graph/rules/rules_mo/three_hop_rule_dict.bin","rb") as f:
    #     fin = torch.load(f)

    # with open("/home/luzhang/Desktop/luzhang/2019fall/Project/kb_graph/rules/rules_mo/two_hop_rule_dict.bin","rb") as f:
    #     fin2 = torch.load(f)

    # rules = list(fin['/tv/tv_program/languages'].keys())
    # # + list(fin2['/tv/tv_program/languages'].keys())

    for ii in range(len(rules)):
        rules[ii] = list(rules[ii])
    print(rules[0])

    tasks = list()
#     with open("/home/luzhang/Desktop/luzhang/2019fall/Project/kb_graph/data/relation_cluster.json","r") as f:
#         edge_cluster = json.load(f)
#     relations = list()
#     for k in sorted(edge_cluster,key = lambda k: len(edge_cluster[k]),reverse = False):
#         relations.append(k)
# #print(relations[d])
#     relations[d] = "/tv/tv_program/languages"
#     #relations[d] = "/location/location/time_zones"
#     # if(relations[d] in rules):
#        # print(rules[relations[d]])
#        # print(rules[relations[d]])
    pair_rel = []
    with open("/home/luzhang/Desktop/luzhang/2019fall/Project/kb_graph/fb15k_237/test.txt","r") as f:
        for line in f:

            line_list = line.strip('\n').split('\t')
            if(line_list[1] =="/tv/tv_program/languages"):
                pair_rel.append((line_list[0],line_list[2],line_list[1]))

    # for item in edge_cluster[relations[d]][4:]:
    for item in pair_rel:
        print(item)
        # x = list(rules[relations[d]])
       # print(tuple(item))
        tasks.append((G,rules,tuple(item),path))
    # else:
    #     sys.exit(0)
    num_cores = multiprocessing.cpu_count()
    pool = multiprocessing.Pool(processes = num_cores-1)

    inputs = tqdm(tasks)

    processed_list = pool.starmap(check_connected_subgraph2,inputs)
    count =0
    node = 0
    size = 0
    ll = 0
    for ii in processed_list:
        if ii:
            ll +=ii[0]
            count +=ii[1]
            size +=ii[2]
            node += ii[3]

    print('count', count, 'pairs', len(edges_dev))
    print('average coverage', count/ll)
    print('average size', size/ll)
    print('average size', node/ll)






