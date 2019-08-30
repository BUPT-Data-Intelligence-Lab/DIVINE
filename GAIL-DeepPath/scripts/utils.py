# -*- coding: utf-8 -*-
from __future__ import division
import random
from collections import namedtuple, Counter
import numpy as np

from BFS.KB import KB
from BFS.BFS import BFS

# hyper-parameters
state_dim = 200
action_space = 400
eps_start = 1
eps_end = 0.1
epe_decay = 1000
replay_memory_size = 10000
batch_size = 128
embedding_dim = 100
gamma = 0.99
target_update_freq = 1000
max_steps = 50
max_steps_test = 50

dataPath = '../NELL-995/'

# define a namedtuple class Transition with the following four properties
Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))


# some convenient utilities
def distance(e1, e2):
    return np.sqrt(np.sum(np.square(e1 - e2)))


def compare(v1, v2):
    return sum(v1 == v2)


def teacher(e1, e2, env, path=None, random_mechanism=True,num_paths=5):  # demo_paths is a list for str(demo paths)
    f = open(path)
    content = f.readlines()
    f.close()
    kb = KB()
    for line in content:
        # rsplit() is from right to left
        ent1, rel, ent2 = line.rsplit()
        kb.addRelation(ent1, rel, ent2)


    # print 'demo_paths:',demo_paths

    res_entity_lists_new = []
    res_path_lists_new = []
    if random_mechanism is True:
        path_str = False  # path_str in the condition is useless
        intermediates = kb.pickRandomIntermediatesBetween(e1, e2, num_paths)
        for i in xrange(num_paths):
            try:
                suc1, entity_list1, path_list1 = BFS(kb, e1, intermediates[i])
                suc2, entity_list2, path_list2 = BFS(kb, intermediates[i], e2)
                if suc1 and suc2:
                    entity_list = entity_list1 + entity_list2[1:]
                    path_list = path_list1 + path_list2
                    res_entity_lists_new.append(entity_list)
                    res_path_lists_new.append(path_list)
            except Exception as e:
                # print'Training Sample:', e1 + ' ' + e2
                print'Cannot find a path'

        if len(res_path_lists_new) == 0:
            print'Cannot find a path'
            return False,False
        else:
            print 'BFS found paths:', len(res_path_lists_new)
    else:
        try:
            suc, entity_list, path_list = BFS(kb, e1, e2)
            path_str = ' -> '.join(path_list)
            # if path_str not in demo_paths:
            #     print 'Not in demo paths'
            #     return False
        except Exception as e:
            # print'Training Sample:', e1 + ' ' + e2
            print'Cannot find a path'
            return False, False
        res_entity_lists_new.append(entity_list)
        res_path_lists_new.append(path_list)
    # path_str = ' -> '.join(path_list)
    # print path_str
    # if path_str not in demo_paths:
    #     print 'Not in demo paths'
    #     return False

    # res_entity_lists_new.append(entity_list)
    # res_path_lists_new.append(path_list)

    # print 'entity_lists:\n', res_entity_lists_new
    # print 'path_lists(rel_lists):\n', res_path_lists_new

    good_episodes = []
    # we need the environment here
    targetID = env.entity2id_[e2]
    for path in zip(res_entity_lists_new, res_path_lists_new):
        good_episode = []
        for i in xrange(len(path[0]) - 1):
            currID = env.entity2id_[path[0][i]]
            nextID = env.entity2id_[path[0][i + 1]]
            state_curr = [currID, targetID, 0]
            state_next = [nextID, targetID, 0]
            actionID = env.relation2id_[path[1][i]]
            # set (state,action,next_state,reward)
            good_episode.append(
                Transition(state=env.idx_state(state_curr), action=actionID, next_state=env.idx_state(state_next),
                           reward=1))
        good_episodes.append(good_episode)
    # print 'good_episodes[0]:\n',good_episodes[0]
    return good_episodes, path_str


def path_clean(path):
    rel_ents = path.split(' -> ')
    relations = []
    entities = []
    for idx, item in enumerate(rel_ents):
        if idx % 2 == 0:
            relations.append(item)
        else:
            entities.append(item)
    entity_stats = Counter(entities).items()
    duplicate_ents = [item for item in entity_stats if item[1] != 1]
    duplicate_ents.sort(key=lambda x: x[1], reverse=True)
    for item in duplicate_ents:
        ent = item[0]
        ent_idx = [i for i, x in enumerate(rel_ents) if x == ent]
        if len(ent_idx) != 0:
            min_idx = min(ent_idx)
            max_idx = max(ent_idx)
            if min_idx != max_idx:
                rel_ents = rel_ents[:min_idx] + rel_ents[max_idx:]
    return ' -> '.join(rel_ents)


def prob_norm(probs):
    return probs / sum(probs)


if __name__ == '__main__':
    print prob_norm(np.array([1, 1, 1]))
