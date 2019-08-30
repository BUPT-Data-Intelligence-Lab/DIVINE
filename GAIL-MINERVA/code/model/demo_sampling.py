from __future__ import division
import tensorflow as tf
import numpy as np
import collections
from itertools import count
import sys

# from utils import *
from BFS.KB import KB
from BFS.BFS import BFS
import time

dataPath = '../../datasets/data_preprocessed/nell-995/'
graphPath = dataPath + 'graph.txt'
relationPath = dataPath + 'train.txt'


def sampling(path_threshold=2, path=None):
    f = open(path)
    content = f.readlines()
    f.close()
    kb = KB()
    for line in content:
        # rsplit() is from right to left
        ent1, rel, ent2 = line.rsplit()
        kb.addRelation(ent1, rel, ent2)

    f = open(relationPath)
    train_data = f.readlines()  # positive sample(h,r,t) from this task in KG
    f.close()

    num_samples = len(train_data)
    demo_path_dict = {}
    for episode in range(num_samples):
        # print "Episode %d" % episode
        # print 'Training Sample:', train_data[episode % num_samples][:-1]  # del the '\n' in the last position
        sample = train_data[episode % num_samples].split()
        ent1 = sample[0]
        ent2 = sample[2]
        rel = sample[1]

        # print(sample[0])
        # print(sample[2])
        # curPath = kb.getPathsFrom(sample[0])
        # print(curPath)

        # temporarily remove the current triple(ent1,rel,ent2)
        # if not, we can only get the current rel as the current path
        kb.removePath(ent1, ent2)
        try:
            suc, entity_list, path_list = BFS(kb, ent1, ent2)
            # if len(path_list) > 1:
            #     print('path_list:\n', len(path_list))
            path_str = ' -> '.join(path_list)
        except Exception as e:
            print('Episode %d' % episode)
            # print('Training Sample:', train_data[episode % num_samples][:-1])  # del the '\n' in the last position
            print('Cannot find a path')
            continue

        if path_str not in demo_path_dict:
            demo_path_dict[path_str] = 1
        else:
            demo_path_dict[path_str] += 1

        if rel not in demo_path_dict:
            demo_path_dict[rel] = 1
        else:
            demo_path_dict[rel] += 1

        # add the current triple back
        kb.addRelation(ent1, rel, ent2)

    # The path has been found at least path_threshold times
    demo_path_dict = {k: v for k, v in demo_path_dict.items() if v >= path_threshold}
    demo_path_list = sorted(demo_path_dict.items(), key=lambda x: x[1], reverse=True)
    # print'demo_path_list:\n', demo_path_list
    print('BFS found paths:', len(demo_path_list))

    f = open(dataPath + 'demo_path.txt', 'w')
    for item in demo_path_list[:5]:
        f.write(item[0] + '\n')
    f.close()
    print('demo path saved')

    f = open(dataPath + 'demo_path_stat.txt', 'w')
    for item in demo_path_list:
        f.write(item[0] + '\t' + str(item[1]) + '\n')
    f.close()
    print('demo path stat saved')

    return


if __name__ == '__main__':
    sampling(path_threshold=2, path=graphPath)
