import numpy as np
import random
from utils import *

# add get_valid_actions_inv


class Env(object):
    """knowledge graph environment definition"""

    def __init__(self, dataPath, task=None):
        f1 = open(dataPath + 'entity2id.txt')
        f2 = open(dataPath + 'relation2id.txt')
        self.entity2id = f1.readlines()
        self.relation2id = f2.readlines()
        f1.close()
        f2.close()
        self.entity2id_ = {}
        self.relation2id_ = {}
        self.relations = []
        for line in self.entity2id:
            # dict{entity:id}
            self.entity2id_[line.split()[0]] = int(line.split()[1])
        for line in self.relation2id:
            # dict{relation:id}
            self.relation2id_[line.split()[0]] = int(line.split()[1])
            # list[all relations]
            self.relations.append(line.split()[0])
        self.entity2vec = np.loadtxt(dataPath + 'entity2vec.bern')  # TransR
        self.relation2vec = np.loadtxt(dataPath + 'relation2vec.bern')  # TransR

        self.path = []
        self.path_relations = []

        # KG used to build the RL environment for path finding
        # raw.kb + inv_relation addition => kb_env_rl
        # so the scale of kb_env_rl is double than raw.kb
        f = open(dataPath + 'kb_env_rl.txt')
        kb_all = f.readlines()
        f.close()

        # need to improvement here !
        # the current method build the self.kb too frequently
        self.kb = []
        if task is not None:
            relation = task.split()[2]
            for line in kb_all:
                rel = line.split()[2]
                if rel != relation and rel != relation + '_inv':
                    self.kb.append(line)

        # print 'len(kb):', len(self.kb)
        self.die = 0  # record how many times does the agent choose an invalid path

    def interact(self, state, action):
        '''
        This function process the interact from the agent
        state: is [current_position, target_position]
        action: an integer
        return: (reward, [new_postion, target_position], done)
        '''
        done = 0  # Whether the episode has finished
        curr_pos = state[0]
        target_pos = state[1]
        chosed_relation = self.relations[action]
        choices = []
        for line in self.kb:
            triple = line.rsplit()
            e1_idx = self.entity2id_[triple[0]]

            if curr_pos == e1_idx and triple[2] == chosed_relation and triple[1] in self.entity2id_:
                choices.append(triple)
        if len(choices) == 0:
            reward = -1
            self.die += 1
            next_state = state  # stay in the initial state
            next_state[-1] = self.die
            return (reward, next_state, done)
        else:  # find a valid step
            path = random.choice(choices)
            self.path.append(path[2] + ' -> ' + path[1])
            self.path_relations.append(path[2])
            # print 'Find a valid step', path
            # print 'Action index', action
            self.die = 0
            new_pos = self.entity2id_[path[1]]
            reward = 0
            new_state = [new_pos, target_pos, self.die]

            if new_pos == target_pos:
                print 'Find a path:', self.path
                done = 1
                reward = 0
                new_state = None
            return (reward, new_state, done)

    def idx_state(self, idx_list):  # idx_list = [currID, targetID, 0]
        if idx_list != None:
            curr = self.entity2vec[idx_list[0], :]  # : represents 'entity2vec.bern'
            targ = self.entity2vec[idx_list[1], :]
            # state s_t = (e_t, e_target-e_t)
            return np.expand_dims(np.concatenate((curr, targ - curr)), axis=0)
        else:
            return None

    def get_valid_actions(self, entityID):
        actions = set()
        for line in self.kb:
            triple = line.split()
            e1_idx = self.entity2id_[triple[0]]
            if e1_idx == entityID:
                actions.add(self.relation2id_[triple[2]])
        return np.array(list(actions))

    def get_valid_actions_inv(self, entityID):
        actions = set()
        for line in self.kb:
            triple = line.split()
            e2_idx = self.entity2id_[triple[1]]
            if e2_idx == entityID:
                actions.add(self.relation2id_[triple[2]])
        return np.array(list(actions))

    def path_embedding(self, path):
        embeddings = [self.relation2vec[self.relation2id_[relation], :] for relation in path]
        embeddings = np.reshape(embeddings, (-1, embedding_dim))
        # simple sum? In the viewpoint of Trans, e_begin+r1+r2+...+rn=e_target => e_begin+path_embedding=e_target
        path_encoding = np.sum(embeddings, axis=0)
        return np.reshape(path_encoding, (-1, embedding_dim))
