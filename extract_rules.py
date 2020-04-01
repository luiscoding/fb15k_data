import networkx as nx
from tqdm import tqdm
import multiprocessing
from datetime import datetime
import collections
import os
from collections import Counter
import jsonlines


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

def rules_extract(edges,G,pair_dev,path):
    rules_dict ={}
    with jsonlines.open(path+'/test_train.jsonl', mode='a') as writer:
        with jsonlines.open(path+'/attr_test_train.jsonl', mode='a') as attr_writer:
            for rel in [pair_dev]:
                positive_samples = set([(u,v) for (u,v,k) in edges if k == rel])
                rules = list()
                # t2 = time.time()
                for k in positive_samples:
                    paths = list(all_simple_paths(G, k[0], k[1],cutoff =4))
                    for pp in  paths:
                        rl = []
                        for ii in pp[1:]:
                            rl.append(ii[1])
                        rules.append(rl)

                # t3 = time.time()
                new_rules = list(map(tuple, rules))
                res = stringify_keys(Counter(new_rules))
                rules_dict[rel] = res
                num_rules = len(Counter(new_rules))
                out_attr = stringify_keys({str(rel):{"number of rules":num_rules,"average frequency":len(rules)/(1+num_rules)}})
                out_res = stringify_keys({str(rel):{"rules":res}})
                attr_writer.write(out_attr)
                writer.write(out_res)

    return True

def construct_original_graph(graph_path):
    head_nodes = list()
    tail_nodes = list()
    edges = list()
    rel_dict = dict()
    # read train data
    with open (graph_path) as fin:
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

    with open ("../kb_graph/fb15k_237/valid.txt") as fin:
        for line in fin:
            line_list = line.strip('\n').split('\t')
            head_nodes_dev.append(line_list[0])
            tail_nodes_dev.append(line_list[2])
            edges_dev.append((line_list[0],line_list[2],line_list[1]))
            rel_dict_dev[(line_list[0],line_list[2])] = line_list[1]
            rell.add(line_list[1])
    return G,edges,edges_dev


G,edges,edges_dev = construct_original_graph()

pair_dev = {}
with jsonlines.open("remain_edge_dev.jsonl") as reader:
    for obj in reader:
        pair_dev = obj


path = "rules_dev_"+ datetime.now().strftime('%H_%M_%d_%m_%Y')
access_rights = 0o777

try:
    os.mkdir(path, access_rights)
except:
    print ("Creation of the directory %s failed" % path)
else:
    print ("Successfully created the directory %s" % path)

tasks =[(edges,G,item,path) for item in pair_dev]

num_cores = multiprocessing.cpu_count()
pool = multiprocessing.Pool(processes = num_cores-1)

inputs = tqdm(tasks)

if __name__ == "__main__":

    processed_list = pool.starmap(rules_extract,inputs)
    print(sum(processed_list))




