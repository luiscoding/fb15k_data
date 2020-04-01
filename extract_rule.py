import networkx as nx
from tqdm import tqdm
import collections
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

    return G

def positive_pairs(relation_path):
    pairs = set()
    with open(relation_path) as fin:
        for line in fin:
            h,t,_ = line.strip("\n").split("\t")
            pairs.add((h,t))
    return pairs

def rule_find(graph_path,relation_path,rules_path):
    G = construct_original_graph(graph_path)
    print(G.size())
    f = open(rules_path,"w")
    pairs = positive_pairs(relation_path)
    rules = set()
    for pair in tqdm(pairs):
        if(G.has_node(pair[0]) and G.has_node(pair[1])):
            paths = list(all_simple_paths(G, pair[0], pair[1], cutoff=4))
            for pp in tqdm(paths):
                rl = []
                for ii in pp[1:]:
                    rl.append(ii[1])
                if(tuple(rl) not in rules):
                    print(rl)
                    rules.add(tuple(rl))
                    f.write("\t".join(rl))
                    f.write("\n")
                    f.flush()
    f.close()
    return True





if __name__ =="__main__":
    dataPath = "./FB15k-237/"

    relations = [
        #"@sports@sports_team@sport"
         "people@person@place_of_birth",
         "people@person@nationality"
        #"@film@film@language"
        #"@film@director@film"
        #"@film@director@film"
        #"@film@film@written_by"
        #"@tv@tv_program@languages"
        #"@location@capital_of_administrative_division@capital_of.@location@administrative_division_capital_relationship@administrative_division" \
        #"@organization@organization_founder@organizations_founded" \
       # "@music@artist@origin" \
        ]
    for relation in relations:
        graphpath = dataPath + 'tasks/' + relation + '/' + 'graph.txt'
        relationPath = dataPath + 'tasks/' + relation + '/' + 'train_pos'
        rules_path = dataPath + 'tasks/' + relation + '/' + 'rules.txt'
        rule_find(graphpath,relationPath,rules_path)









