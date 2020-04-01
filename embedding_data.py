
import numpy as np
import sys
import random

def rewriteFile(relationPath):
    fout = open(relationPath + "_path_filtered_train_deep", "w")
    with open(relationPath+"_path_filtered_train","r") as f:
        ff = f.readlines()
        for line in ff:
            print(line.split("\t"))
            fout.write("&".join(line.split("\t")))
    fout.close()

def rulesEmbedding(dataPath,relation):
    embedding_rules = []
    headpath2id = open(dataPath+"headpath2id_nohead"+relation+".txt","w")
    headpath2vec = open(dataPath+"embedding_paths_new_train_nohead"+relation+".npy", "wb")
    with open(dataPath+"rules.txt", "r")as f:
        lines = f.readlines()
        for line in lines:
            embedding_rules.append(line.strip().split("\t"))

    relation_id_dict = {}
    with open(dataPath +'relation2id.txt', "r") as f1:
        for line in f1.readlines():
            rel, idx = line.split("\t")
            relation_id_dict[rel] = idx

    rel_vec = np.loadtxt(dataPath + '/relation2vec.unif')

    count = 0
    headpath2id.write("@".join("") + "&" + str(count)+"\n")
    count = 1
    embedding_res = []
    x_0 = rel_vec[0]
    for ii in range(len(x_0)):
        x_0[ii]=0
    embedding_res.append(x_0)
    for head_rule in embedding_rules:
        headpath2id.write("@".join(head_rule) + "&" + str(count)+"\n")
        count += 1
        x_0 = rel_vec[0]
        for ii in range(len(x_0)):
            x_0[ii] = 0

        for jj in range(0, len(head_rule)):
            #             x = y[relation_id_dict[jj]]
            idx = relation_id_dict[head_rule[jj]]
            x_0 = np.add(rel_vec[int(idx)], x_0)
        embedding_res.append(x_0)

    embedding_matrix = np.array(embedding_res).astype(np.float32)

    print(type(embedding_matrix))
    np.save(headpath2vec, embedding_matrix, allow_pickle=True)

def trainDataSelected(ratio,relationPath,relation):
    train_pos = []
    with open(relationPath+"train_pos","r") as f:
        ff = f.readlines()
        for line in ff:
            h,t,_ = line.strip("\n").split("\t")
            train_pos.append(h+t)
    print(len(train_pos))
    train_data = {}
    train_data_list = []
    with open(relationPath + "train.pairs_path_filtered_train", "r") as f:
        ff = f.readlines()
        for line in ff:
            x = line.strip().strip("\n").split("\t")
            if(len(x)>4):
                print(x[1])
            lbl= x[0]
            h = x[-2]
            t = x[-1]
            txt = " ".join(x[1:-2])
            train_data[h+t] = (lbl,txt,h,t)
            train_data_list.append(((lbl,txt,h,t)))
    sort_train = sorted(train_data_list,key =lambda t:(t[2],t[0]),reverse=True)
    out_train = []
    ii =0
    while ii < len(sort_train):
        if sort_train[ii][0] =="1":
            for idx in range(ratio):
                if(ii+idx<len(sort_train)):
                    out_train.append(sort_train[ii+idx])
            ii += 4
        ii+=1


    print(len(out_train))
    random.shuffle(out_train)
    sorted_out = sorted(out_train,key =lambda t:t[2])
    with open(relationPath+"training_data_path_train_nohead"+relation+".txt","w") as fout:
        for ii in sorted_out:
            fout.write(str("&".join(list(ii)))+"\n")

def reorderTest(relationPath,relation):
    train_pairs = list()
    with open(relationPath+"sort_test.pairs", "r") as fdeep:
        fin = fdeep.readlines()
        for line in fin:
            lbl = line.strip("\n").split(":")[-1].strip()
            h, t = line.strip("\n").split(":")[0].strip().split(",")
            h = h.split("$")[1]
            t = t.split("$")[1]
            train_pairs.append((h, t,lbl))

    train_read = dict()
    with  open(relationPath + "sort_test.pairs_path_filtered_test", "r") as ffilter:
        fin = ffilter.readlines()
        for line in fin:
            lbl,txt,h,t = line.strip("\n").split("&")
            train_read[(h, t)] = [lbl,txt,h,t]

    with open(relationPath+"testing_data_path_train_nohead"+relation+".txt","w") as fout:
        for key in train_pairs:
            if((key[0], key[1]) in train_read):
                if key[2] =="+":
                    fout.write((str(1) + "&" + train_read[(key[0], key[1])][1] + "&" + key[0] + "&" + key[1] + "\n"))
                    fout.flush()
                else:
                    fout.write((str(0) + "&" + train_read[(key[0], key[1])][1]+ "&" + key[0] + "&" + key[1] + "\n"))
                    fout.flush()
            else:
                if key[2] =="+":
                    fout.write((str(1) + "&" + " ".join([]) + "&" + key[0] + "&" + key[1] + "\n"))
                    fout.flush()
                else:
                    fout.write((str(0) + "&" + " ".join([]) + "&" + key[0] + "&" + key[1] + "\n"))
                    fout.flush()


def statistic(relationPath,relation):
    count = 0
    with open(relationPath + "testing_data_path_train_nohead" + relation + ".txt", "r") as ff:
        fin = ff.readlines()
        for line in fin:
            x = line.strip("\n").split("&")
            if(len(x)>4):
                print(x[4])
            lbl,txt,h,t = line.strip("\n").split("&")
            count+=len(set(txt.split(" ")))
    print("relation:",relation,count/len(fin))

def validate_deep(relationPath,relation):
    fout = open(relationPath+"sort_test_out.pairs","w")
    with  open(relationPath + "testing_data_path_train_nohead"+relation+".txt","r") as ffilter:
        fin = ffilter.readlines()
        for line in fin:
            lbl,txt,h,t = line.strip("\n").split("&")
            if lbl == "1":
                fout.write("thing$"+h+","+"thing$"+t+": +\n")
                fout.flush()
            else:
                fout.write("thing$" + h + "," + "thing$" + t + ": -\n")
                fout.flush()

    fout.close()


def embedding_onehot():

    headpath2vec = open("embedding_paths_new_onehot.npy", "wb")
    dataPath = "./NELL-995/" + 'tasks/' + "concept_athletehomestadium" + '/'
    relation_id_dict = {}
    with open(dataPath + 'relation2id.txt', "r") as f1:
        for line in f1.readlines():
            rel, idx = line.split("\t")
            relation_id_dict[rel] = idx

    rel_vec = np.loadtxt(dataPath + '/relation2vec.unif')

    embedding_res = []
    x_0 = rel_vec[0]
    for ii in range(len(x_0)):
        x_0[ii] = 0
    embedding_res.append(x_0)

    x_0 = rel_vec[0]
    for ii in range(len(x_0)):
        x_0[ii] = 1
    embedding_res.append(x_0)
    embedding_matrix = np.array(embedding_res).astype(np.float32)

    print(type(embedding_matrix))
    np.save(headpath2vec, embedding_matrix, allow_pickle=True)




if __name__ =="__main__":

    dataPath = "./NELL-995/"
    relations = ["concept_athletehomestadium",
                "concept_athleteplaysforteam",
                "concept_athleteplaysinleague",
                #"concept_athleteplayssport",
                "concept_organizationheadquarteredincity",
                "concept_organizationhiredperson",
                "concept_personborninlocation",
                #"concept_personleadsorganization"]
               "concept_teamplayssport",
               "concept_worksfor"]
    for relation in relations:
        graphpath = dataPath + 'tasks/' + relation + '/' + 'graph.txt'
        relationPath = dataPath + 'tasks/' + relation + '/'
        rulesPath = dataPath + 'tasks/' + relation + '/' + 'rules.txt'

        #trainDataSelected(4,relationPath,relation)

        #sort_train_thing(relationPath)
       # path_train_data(relationPath,graphpath,rulesPath)
       #
        rulesEmbedding(dataPath+ 'tasks/' + relation +'/',relation)
       # reorderTest(relationPath, relation)
       # statistic(relationPath, relation)
       # validate_deep(relationPath,relation)
    #embedding_onehot()
    #sort_deep(relationPath)