#!/usr/bin/python
# dot att

import numpy as np
import sys
from env import Env
from utils import *
from BFS.KB import *

# print sys.argv
relation = sys.argv[1]
task = relation.replace('_', ':', 1)
print('task:', task)

dataPath_ = '../NELL-995/tasks/' + relation
featurePath = dataPath_ + '/path_to_use_total.txt'
feature_stats = dataPath_ + '/path_stats_test.txt'
relationId_path = '../NELL-995/' + 'relation2id.txt'
ent_id_path = '../NELL-995/' + 'entity2id.txt'
rel_id_path = '../NELL-995/' + 'relation2id.txt'
test_data_path = '../NELL-995/tasks/' + relation + '/sort_test.pairs'
fact_results_path = '../NELL-995/results' + '/fact_results.txt'


def bfs_two(e1, e2, path, kb, kb_inv):
    start = 0
    end = len(path)
    left = set()
    right = set()
    left.add(e1)
    right.add(e2)

    left_path = []
    right_path = []
    while (start < end):
        left_step = path[start]
        left_next = set()
        right_step = path[end - 1]
        right_next = set()

        if len(left) < len(right):
            left_path.append(left_step)
            start += 1
            for entity in left:
                try:
                    for path_ in kb.getPathsFrom(entity):
                        if path_.relation == left_step:
                            left_next.add(path_.connected_entity)
                except Exception as e:
                    # print 'left', len(left)
                    # print left
                    # print 'not such entity'
                    return False
            left = left_next

        else:
            right_path.append(right_step)
            end -= 1
            for entity in right:
                try:
                    for path_ in kb_inv.getPathsFrom(entity):
                        if path_.relation == right_step:
                            right_next.add(path_.connected_entity)
                except Exception as e:
                    # print 'right', len(right)
                    # print 'no such entity'
                    return False
            right = right_next

    if len(right & left) != 0:
        return True
    return False


def get_features():  # a little different with get_features() in link prediction
    stats = {}
    with open(feature_stats) as f:
        path_freq = f.readlines()


    if len(path_freq) > 0:
        for line in path_freq:
            path = line.split('\t')[0]
            num = int(line.split('\t')[1])
            stats[path] = num
    else:
        with open(featurePath) as file:
            path_freq = file.readlines()
        for line in path_freq:
            path = line.split('\t')[0]
            num = 1
            stats[path] = num



    max_freq = np.max(stats.values())

    relation2id = {}
    f = open(relationId_path)
    content = f.readlines()
    f.close()
    for line in content:
        relation2id[line.split()[0]] = int(line.split()[1])

    relation2vec = np.loadtxt(dataPath + 'relation2vec.bern')
    embed_task = relation2vec[relation2id[task]]
    print('task id:', relation2id[task])
    # print('task embedding shape:', embed_task.shape)
    # print('task embedding:',embed_task)

    env = Env(dataPath)
    f = open(featurePath)
    paths = f.readlines()
    embed_paths = np.array([env.path_embedding(ind.strip().rsplit(' -> ')) for ind in paths])
    embed_paths = embed_paths.astype('float32')
    embed_paths = np.squeeze(embed_paths)  # n*100
    f.close()
    # print('embed_paths:',embed_paths.shape)

    # if len(paths) > 20:
    #     paths = paths[:20]

    weights = []
    for i in range(len(paths)):
        w = np.dot(embed_task, embed_paths[i])
        weights.append(w)

    weights = 1/(1 + 1/(np.exp(weights)))
    # print weights
    # raw_input('----------')
    # sum = np.sum(weights)
    # print weights
    # print 'sum:', sum
    # weights = weights / sum
    # sum = np.sum(weights)
    # print weights
    # print 'sum:', sum
    # max_dist = np.max(weights)
    # min_dist = np.min(weights)
    # weights = 1-(weights - min_dist)/(max_dist-min_dist)

    # print(weights)


    # dist = np.linalg.norm(np.zeros(100) - embed_task)
    # print dist

    # print paths
    # print
    paths_stats = list(zip(paths, weights))

    # print stats
    # print paths
    # paths from path_to_use while stats from path_stats
    useful_paths = []
    named_paths = []
    for line in paths_stats:
        # if line[1] > 1:
        #     continue
        path = line[0].rstrip()
        if path not in stats:  # why path must in stats
             continue
        elif max_freq > 1 and stats[path] < 2:
             continue

        length = len(path.split(' -> '))
        if length <= 10:
            pathIndex = []
            pathName = []
            relations = path.split(' -> ')

            for rel in relations:
                pathName.append(rel)
                rel_id = relation2id[rel]
                pathIndex.append(rel_id)
            useful_paths.append(pathIndex)
            named_paths.append((pathName, line[1]))

    print 'How many paths used: ', len(useful_paths)
    # print 'useful_paths:', useful_paths
    # print 'named_paths:', named_paths
    return useful_paths, named_paths


f1 = open(ent_id_path)
f2 = open(rel_id_path)
content1 = f1.readlines()
content2 = f2.readlines()
f1.close()
f2.close()

entity2id = {}
relation2id = {}
for line in content1:
    entity2id[line.split()[0]] = int(line.split()[1])

for line in content2:
    relation2id[line.split()[0]] = int(line.split()[1])

_, named_paths = get_features()
path_weights = []
for item in named_paths:
    path = item[0]
    weight = 1.0 / len(path)
    path_weights.append(weight)
path_weights = np.array(path_weights)
kb = KB()
kb_inv = KB()

f = open(dataPath_ + '/graph.txt')
kb_lines = f.readlines()
f.close()

for line in kb_lines:
    e1 = line.split()[0]
    rel = line.split()[1]
    e2 = line.split()[2]
    kb.addRelation(e1, rel, e2)
    kb_inv.addRelation(e2, rel, e1)

f = open(test_data_path)  # '../NELL-995/tasks/'  + relation + '/sort_test.pairs'
test_data = f.readlines()
f.close()
test_pairs = []
test_labels = []
test_set = set()
for line in test_data:
    e1 = line.split(',')[0].replace('thing$', '')
    # e1 = '/' + e1[0] + '/' + e1[2:]
    e2 = line.split(',')[1].split(':')[0].replace('thing$', '')
    # e2 = '/' + e2[0] + '/' + e2[2:]
    # if (e1 not in kb.entities) or (e2 not in kb.entities):
    #	continue
    test_pairs.append((e1, e2))
    label = 1 if line[-2] == '+' else 0
    test_labels.append(label)

aps = []
query = test_pairs[0][0]
y_true = []
y_score = []
hit_1_list = []
hit_3_list = []
hit_10_list = []
mrr_list = []

score_all = []

for idx, sample in enumerate(test_pairs):
    # print 'query node: ', sample[0], idx
    if sample[0] == query:
        features = []
        for p in named_paths:
            path = p[0]
            features.append(1 * int(bfs_two(sample[0], sample[1], path, kb, kb_inv)))
        score = sum(features)  # maybe we can do some works here
        y_score.append(score)
        y_true.append(test_labels[idx])
    else:  # begin to next test batch
        query = sample[0]
        # print (y_true)
        count = list(zip(y_score, y_true))
        count.sort(key=lambda x: x[0], reverse=True)
        # print ('count:',len(count))

        ranks = []
        correct = 0

        hit_1 = 0
        hit_3 = 0
        hit_10 = 0
        mrr = 0

        # almost every count only have correct item
        # because in sort_test.pairs almost 1+ with several - for every test_pair
        for idx_, item in enumerate(count):
            if item[1] == 1:
                correct += 1
                ranks.append(correct / (1.0 + idx_))

                # only use the first positive sample to evaluate hits@n
                if correct == 1:
                    if idx_ < 10:
                        hit_10 += 1
                        if idx_ < 3:
                            hit_3 += 1
                            if idx_ < 1:
                                hit_1 += 1
                if mrr == 0:
                    mrr = 1/(1.0 + idx_)

        if len(ranks) == 0:
            aps.append(0)
        else:
            aps.append(np.mean(ranks))

        hit_1_list.append(hit_1)
        hit_3_list.append(hit_3)
        hit_10_list.append(hit_10)
        if correct == 0:
            mrr_list.append(0)
        else:
            mrr_list.append(mrr / correct)
        y_true = []
        y_score = []
        features = []
        for p in named_paths:
            path = p[0]
            features.append(1 * int(bfs_two(sample[0], sample[1], path, kb, kb_inv)))
        score = sum(features)  # maybe we can do some works here
        y_score.append(score)
        y_true.append(test_labels[idx])

count = list(zip(y_score, y_true))
count.sort(key=lambda x: x[0], reverse=True)
# print ('count:',count)

ranks = []
correct = 0

hit_1 = 0
hit_3 = 0
hit_10 = 0
mrr = 0

for idx_, item in enumerate(count):
    if item[1] == 1:
        correct += 1
        ranks.append(correct / (1.0 + idx_))

        # only use the first positive sample to evaluate hits@n
        if correct == 1:
            if idx_ < 10:
                hit_10 += 1
                if idx_ < 3:
                    hit_3 += 1
                    if idx_ < 1:
                        hit_1 += 1
        if mrr == 0:
            mrr = 1 / (1.0 + idx_)

aps.append(np.mean(ranks))

hit_1_list.append(hit_1)
hit_3_list.append(hit_3)
hit_10_list.append(hit_10)
if correct == 0:
    mrr_list.append(0)
else:
    mrr_list.append(mrr / correct)

# print hit_10_list

mean_ap = np.mean(aps)
mean_hit_1 = np.mean(hit_1_list)
mean_hit_3 = np.mean(hit_3_list)
mean_hit_10 = np.mean(hit_10_list)
mean_mrr = np.mean(mrr_list)

print 'HITS@1: ', mean_hit_1
print 'HITS@3: ', mean_hit_3
print 'HITS@10: ', mean_hit_10
print 'MRR: ', mean_mrr
print 'MAP: ', mean_ap
#
# with open(fact_results_path, 'a') as f:
#     f.write(relation + ':\n')
#     f.write('RL MAP: ' + str(mean_ap) + '\n' + 'HITS@1: ' + str(mean_hit_1) + '\n' + 'HITS@3: ' + str(
#         mean_hit_3) + '\n' + 'HITS@10: ' + str(mean_hit_10) + '\n' + 'MRR: ' + str(mean_mrr) + '\n')

# ranking all positive and negative samples
scores_rl = []
query = test_pairs[0][0]
# print 'How many queries: ', len(test_pairs)
for idx, sample in enumerate(test_pairs):
    # print 'Query No.%d of %d' % (idx, len(test_pairs))
    # RL
    features = []
    for item in named_paths:
        path = item[0]
        alpha = item[1]
        # alpha = np.exp(-item[1])
        # alpha = 1
        features.append(alpha * int(bfs_two(sample[0], sample[1], path, kb, kb_inv)))
    # features = features*path_weights
    score_rl = sum(features)  # maybe we can do some works here
    scores_rl.append(score_rl)

rank_stats_rl = zip(scores_rl, test_labels)
rank_stats_rl.sort(key=lambda x: x[0], reverse=True)


correct = 0
ranks = []

# print rank_stats_rl
for idx, item in enumerate(rank_stats_rl):
    if item[1] == 1:
        correct += 1
        ranks.append(correct / (1.0 + idx))
mean_ap_total = np.mean(ranks)


print 'MAP*: ', mean_ap_total

with open(fact_results_path, 'a') as f:
    f.write(relation + '\n' + 'Hits1 ' + 'Hits3 ' + 'MRR ' + 'MAP ' + 'MAP* ' + '\n')
    f.write(str(mean_hit_1) + '\t' + str(mean_hit_3) + '\t' + str(mean_mrr) + '\t' + str(mean_ap) + '\t' + str(mean_ap_total) + '\n')
