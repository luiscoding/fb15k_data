import cPickle
import sys
import numpy as np

relation = sys.argv[1]

dataPath_ = './tasks/'  + relation

ent_id_path = dataPath_ + '/entity2id.txt'
rel_id_path = dataPath_ + '/relation2id.txt'
test_data_path = './tasks/'  + relation + '/sort_test.pairs'

f1 = open(ent_id_path)
f2 = open(rel_id_path)
content1 = f1.readlines()
content2 = f2.readlines()
f1.close()
f2.close()

entity2id = {}
relation2id = {}
for line in content1[1:]:
	entity2id[line.split()[0]] = int(line.split()[1])

for line in content2[1:]:
	relation2id[line.split()[0]] = int(line.split()[1])


ent_vec = np.loadtxt('./entity2vec.bern')
rel_vec = np.loadtxt('./relation2vec.bern')

f = open(test_data_path)
test_data = f.readlines()
f.close()

test_pairs = []
test_labels = []
# queries = set()
for line in test_data:
	e1 = line.split(',')[0].replace('thing$','')
	e1 = '/' + e1[0] + '/' + e1[2:]
	e2 = line.split(',')[1].split(':')[0].replace('thing$','')
	e2 = '/' + e2[0] + '/' + e2[2:]
	test_pairs.append((e1,e2))
	label = 1 if line[-2] == '+' else 0
	test_labels.append(label)


aps = []
query = test_pairs[0][0]
y_true = []
y_score = []
query_samples = []

score_all = []

rel = '/' + relation.replace("@", "/")
relation_vec = rel_vec[relation2id[rel],:]

g = open(dataPath_ + '/topk.pairs','w')

for idx, sample in enumerate(test_pairs):
	#print 'query node: ', sample[0], idx
	if sample[0] == query:
		e1_vec = ent_vec[entity2id[sample[0]],:]
		e2_vec = ent_vec[entity2id[sample[1]],:]
		score = -np.sum(np.square(e1_vec + relation_vec - e2_vec))
		score_all.append(score)
		y_score.append(score)
		y_true.append(test_labels[idx])
		query_samples.append(sample)
	else:
		query = sample[0]
		count = zip(y_score, y_true, query_samples)
		count.sort(key = lambda x:x[0], reverse=True)
		for idx_, item in enumerate(count):
			if idx_ <= 40:
				g.write(item[2][0]+'\t'+item[2][1]+'\t'+str(item[1])+'\n')

		ranks = []
		correct = 0
		for idx_, item in enumerate(count):
			if item[1] == 1:
				correct +=  1
				ranks.append(correct/(1.0+idx_))
				#break
		if len(ranks)==0:
			ranks.append(0)
		#print np.mean(ranks)
		aps.append(np.mean(ranks))
		if len(aps) % 10 == 0:
			print 'How many queries:', len(aps)
			print np.mean(aps)
		y_true = []
		y_score = []
		query_samples = []
		e1_vec = ent_vec[entity2id[sample[0]],:]
		e2_vec = ent_vec[entity2id[sample[1]],:]

		score = -np.sum(np.square(e1_vec + relation_vec - e2_vec))
		score_all.append(score)
		y_score.append(score)
		y_true.append(test_labels[idx])
		query_samples.append(sample)

g.close()
score_label = zip(score_all, test_labels)
score_label_ranked = sorted(score_label, key = lambda x:x[0], reverse=True)

hits = 0 
for idx, item in enumerate(score_label_ranked):
	if item[1] == 1:
		hits += 1
	if idx == 9:
		print 'P@10: ', hits/10.0
	elif idx ==99:
		print 'P@100: ', hits/100.0
		break

mean_ap = np.mean(aps)
print 'MAP: ', mean_ap
