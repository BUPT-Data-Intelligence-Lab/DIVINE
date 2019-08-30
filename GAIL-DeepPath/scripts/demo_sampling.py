from __future__ import division
import tensorflow as tf
import numpy as np
import collections
from itertools import count
import sys

from utils import *
from BFS.KB import KB
from BFS.BFS import BFS
import time

relation = sys.argv[1]
# dataPath = '../NELL-995/' was settled in utils.py
graphpath = dataPath + 'tasks/' + relation + '/' + 'graph.txt'
relationPath = dataPath + 'tasks/' + relation + '/' + 'train_pos'


def sampling(path_threshold=1, path=None):
    f = open(path)
    content = f.readlines()
    f.close()
    kb = KB()
    for line in content:
        # rsplit() is from right to left
        ent1, rel, ent2 = line.rsplit()
        kb.addRelation(ent1, rel, ent2)

    f = open(relationPath)
    train_data = f.readlines()  # positive sample(h,t,r) from this task in KG
    f.close()

    num_samples = len(train_data)
    demo_path_dict = {}
    for episode in xrange(num_samples):
        # print "Episode %d" % episode
        # print 'Training Sample:', train_data[episode % num_samples][:-1]  # del the '\n' in the last position
        sample = train_data[episode % num_samples].split()

        try:
            suc, entity_list, path_list = BFS(kb, sample[0], sample[1])
            # print'path_list:\n', path_list
            path_str = ' -> '.join(path_list)
        except Exception as e:
            print "Episode %d" % episode
            print 'Training Sample:', train_data[episode % num_samples][:-1]  # del the '\n' in the last position
            print'Cannot find a path'
            continue

        if path_str not in demo_path_dict:
            demo_path_dict[path_str] = 1
        else:
            demo_path_dict[path_str] += 1

    # The path has been found at least path_threshold times
    demo_path_dict = {k: v for k, v in demo_path_dict.iteritems() if v >= path_threshold}
    demo_path_list = sorted(demo_path_dict.items(), key=lambda x: x[1], reverse=True)
    # print'demo_path_list:\n', demo_path_list
    print'BFS found paths:', len(demo_path_list)

    f = open(dataPath + 'tasks/' + relation + '/' + 'demo_path.txt', 'w')
    for item in demo_path_list:
        f.write(item[0]+'\n')
    f.close()
    print 'demo path saved'

    f = open(dataPath + 'tasks/' + relation + '/' + 'demo_path_stat.txt', 'w')
    for item in demo_path_list:
        f.write(item[0]+'\t'+str(item[1])+'\n')
    f.close()
    print 'demo path stat saved'

    return

if __name__ == '__main__':
    sampling(path_threshold=1,path=graphpath)
