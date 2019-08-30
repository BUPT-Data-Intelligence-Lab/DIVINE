#!/usr/bin/python
# add QA evaluation

import sys
import numpy as np
import os
from BFS.KB import *
from sklearn import linear_model
from keras.models import Sequential
from keras.layers import Dense, Activation  # Dense layer means fully connected layer

# print('sys.argv:', sys.argv)
relation = sys.argv[1]

dataPath_ = '../NELL-995/tasks/' + relation
featurePath = dataPath_ + '/path_to_use_total.txt'
# feature_stats = dataPath_ + '/path_stats_test.txt'
relationId_path = '../NELL-995/relation2id.txt'
link_results_path = '../NELL-995/results' + '/link_results.txt'

os.environ["CUDA_VISIBLE_DEVICES"] = "3"


def train(kb, kb_inv, named_paths):  # named_paths are returned by get_features()
    f = open(dataPath_ + '/train.pairs')  # train triples in the PRA format
    train_data = f.readlines()
    f.close()
    train_pairs = []
    train_labels = []
    for line in train_data:
        e1 = line.split(',')[0].replace('thing$', '')
        e2 = line.split(',')[1].split(':')[0].replace('thing$', '')
        if (e1 not in kb.entities) or (e2 not in kb.entities):
            continue
        train_pairs.append((e1, e2))
        label = 1 if line[-2] == '+' else 0
        train_labels.append(label)
    training_features = []
    pairs_num = len(train_pairs)
    for ind, sample in enumerate(train_pairs):
        if ind % 1000 == 0:
            print 'pair num:',ind,'/',pairs_num
        feature = []
        for path in named_paths:
            feature.append(int(bfs_two(sample[0], sample[1], path, kb, kb_inv)))
        training_features.append(feature)
    # print ('training_features:', len(training_features))
    # print ('train_labels:', len(train_labels))
    model = Sequential()
    input_dim = len(named_paths)  # the named_paths found by RL agent decide input_dim and feature_dim
    model.add(Dense(1, activation='sigmoid',
                    input_dim=input_dim))  # output = activation(dot(input, kernel)+bias)  output dimension = 1
    model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(np.array(training_features), np.array(train_labels), epochs=300, batch_size=128, verbose=1)
    return model


def get_features():  # return useful_paths, named_paths
    relation2id = {}
    f = open(relationId_path)  # relation2id.txt
    content = f.readlines()
    f.close()
    for line in content:
        relation2id[line.split()[0]] = int(line.split()[1])

    useful_paths = []
    named_paths = []
    f = open(featurePath)  # dataPath_ + '/path_to_use.txt'
    paths = f.readlines()
    f.close()

    # print(len(paths))
    if len(paths) > 100:
        paths = paths[:100]

    for line in paths:
        path = line.rstrip()

        length = len(path.split(' -> '))

        if length <= 10:  # remain the paths in path_to_use.txt which length is no more than 10
            pathIndex = []
            pathName = []
            relations = path.split(' -> ')

            for rel in relations:
                pathName.append(rel)
                rel_id = relation2id[rel]
                pathIndex.append(rel_id)
            useful_paths.append(pathIndex)
            named_paths.append(pathName)

    print('How many paths used: ', len(useful_paths))
    # print (useful_paths)
    # print (named_paths)
    return useful_paths, named_paths


def evaluate_logic():  # using in main()  return RL MAP
    kb = KB()  # class KB form BFS.KB.py
    kb_inv = KB()  # class KB form BFS.KB.py

    f = open(dataPath_ + '/graph.txt')
    kb_lines = f.readlines()
    f.close()

    for line in kb_lines:
        e1 = line.split()[0]
        rel = line.split()[1]
        e2 = line.split()[2]
        kb.addRelation(e1, rel, e2)
        kb_inv.addRelation(e2, rel, e1)

    _, named_paths = get_features()

    model = train(kb, kb_inv, named_paths)

    f = open(dataPath_ + '/sort_test.pairs')  # sort_test.txt in alphabetical order
    test_data = f.readlines()  # the test data for the whole model
    f.close()
    test_pairs = []
    test_labels = []
    # queries = set()
    for line in test_data:
        e1 = line.split(',')[0].replace('thing$', '')
        # e1 = '/' + e1[0] + '/' + e1[2:]
        e2 = line.split(',')[1].split(':')[0].replace('thing$', '')
        # e2 = '/' + e2[0] + '/' + e2[2:]
        if (e1 not in kb.entities) or (e2 not in kb.entities):
            continue
        test_pairs.append((e1, e2))
        label = 1 if line[-2] == '+' else 0
        test_labels.append(label)

    # print ('test_pairs:',test_pairs)
    # print ('test_labels:',test_labels)
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
            # print 'query:',query
            # print 'sample:',sample[0]
            # print 'y_ture:',y_true
            features = []
            for path in named_paths:
                features.append(int(bfs_two(sample[0], sample[1], path, kb, kb_inv)))

            # score is a np.array([[float32]], dtype=float32))
            score = model.predict(np.reshape(features, [1, -1]))

            # score = np.sum(features)
            # print ('score:',score)
            score_all.append(score[0])
            y_score.append(score)
            y_true.append(test_labels[idx])
        else:  # begin to next test batch
            # print 'query:',query
            # print 'sample:',sample[0]
            # print 'y_ture:', y_true
            # raw_input('----------')
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
                mrr_list.append(mrr/correct)
            # print np.mean(ranks)
            # if len(aps) % 10 == 0:
            # 	print 'How many queries:', len(aps)
            # 	print np.mean(aps)
            y_true = []
            y_score = []
            features = []
            for path in named_paths:
                features.append(int(bfs_two(sample[0], sample[1], path, kb, kb_inv)))  # bfs_two returns True or False

            # features = features*path_weights
            # score = np.inner(features, path_weights)
            # score = np.sum(features)
            score = model.predict(np.reshape(features, [1, -1]))

            score_all.append(score[0])
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

        # if hit_10 > 1:
        #     print count
        #     raw_input('----------')

    # print (ranks)
    aps.append(np.mean(ranks))

    hit_1_list.append(hit_1)
    hit_3_list.append(hit_3)
    hit_10_list.append(hit_10)
    if correct == 0:
        mrr_list.append(0)
    else:
        mrr_list.append(mrr / correct)
    # score_label = zip(score_all, test_labels)
    # score_label_ranked = sorted(score_label, key=lambda x: x[0], reverse=True)
    # print ('score_label_ranked:',len(score_label_ranked))
    # print ('aps:', aps)

    # print hit_10_list

    mean_ap = np.mean(aps)
    mean_hit_1 = np.mean(hit_1_list)
    mean_hit_3 = np.mean(hit_3_list)
    mean_hit_10 = np.mean(hit_10_list)
    mean_mrr = np.mean(mrr_list)
    print 'RL MAP: ', mean_ap
    print 'HITS@1: ', mean_hit_1
    print 'HITS@3: ', mean_hit_3
    print 'HITS@10: ', mean_hit_10
    print 'MRR: ', mean_mrr

    with open(link_results_path, 'a') as f:
        f.write(relation + ':\n')
        f.write('RL MAP: ' + str(mean_ap) + '\n' + 'HITS@1: ' + str(mean_hit_1) + '\n' + 'HITS@3: ' + str(
            mean_hit_3) + '\n' + 'HITS@10: ' + str(mean_hit_10) + '\n' + 'MRR: ' + str(mean_mrr) + '\n')



def bfs_two(e1, e2, path, kb, kb_inv):
    start = 0
    end = len(path)
    left = set()
    right = set()
    left.add(e1)
    right.add(e2)

    left_path = []
    right_path = []
    while start < end:
        left_step = path[start]
        left_next = set()
        right_step = path[end - 1]
        right_next = set()

        if len(left) < len(right):
            left_path.append(left_step)
            start += 1
            # print 'left',start
            # for triple in kb:
            # 	if triple[2] == left_step and triple[0] in left:
            # 		left_next.add(triple[1])
            # left = left_next
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


if __name__ == '__main__':
    evaluate_logic()
