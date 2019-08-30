from __future__ import absolute_import
from __future__ import division
import numpy as np
from code.data.feed_data import RelationEntityBatcher
from code.data.grapher import RelationEntityGrapher
import logging

logger = logging.getLogger()


class Episode(object):

    def __init__(self, graph, data, params):
        self.grapher = graph
        self.batch_size, self.path_len, num_rollouts, test_rollouts, positive_reward, negative_reward, mode, batcher = params
        self.mode = mode

        # default reward are 1 or 0
        # print positive_reward
        # print negative_reward
        # raw_input('----------')

        if self.mode == 'train':
            self.num_rollouts = num_rollouts
        else:
            self.num_rollouts = test_rollouts
        self.current_hop = 0
        start_entities, query_relation, end_entities, all_answers = data

        self.no_examples = start_entities.shape[0]
        self.positive_reward = positive_reward
        self.negative_reward = negative_reward
        # roll out
        start_entities = np.repeat(start_entities, self.num_rollouts)
        batch_query_relation = np.repeat(query_relation, self.num_rollouts)
        end_entities = np.repeat(end_entities, self.num_rollouts)
        self.start_entities = start_entities
        self.end_entities = end_entities
        self.current_entities = np.array(start_entities)
        self.query_relation = batch_query_relation
        self.all_answers = all_answers

        # [batch_size*num_rollouts]
        # default setting is 128*20
        # print self.current_entities.shape
        # print len(self.query_relation)
        # print len(self.all_answers)

        next_actions = self.grapher.return_next_actions(self.current_entities, self.start_entities, self.query_relation,
                                                        self.end_entities, self.all_answers, self.current_hop == self.path_len - 1,
                                                        self.num_rollouts)
        self.state = {}
        self.state['next_relations'] = next_actions[:, :, 1]  # [batch_size*num_rollouts, max_num_actions]
        self.state['next_entities'] = next_actions[:, :, 0]  # [batch_size*num_rollouts, max_num_actions]
        self.state['current_entities'] = self.current_entities

        # print self.state
        # print self.state['next_relations'].shape
        # print self.state['next_entities'].shape
        # print self.state['current_entities'].shape
        # raw_input('----------')

    def get_state(self):
        return self.state

    def get_query_relation(self):
        return self.query_relation

    def get_reward(self):
        reward = (self.current_entities == self.end_entities)

        # set the True and False values to the values of positive and negative rewards.
        condlist = [reward == True, reward == False]
        choicelist = [self.positive_reward, self.negative_reward]
        reward = np.select(condlist, choicelist)  # [B,]
        return reward

    def __call__(self, action):
        self.current_hop += 1
        self.current_entities = self.state['next_entities'][np.arange(self.no_examples*self.num_rollouts), action]

        next_actions = self.grapher.return_next_actions(self.current_entities, self.start_entities, self.query_relation,
                                                        self.end_entities, self.all_answers, self.current_hop == self.path_len - 1,
                                                        self.num_rollouts )

        self.state['next_relations'] = next_actions[:, :, 1]
        self.state['next_entities'] = next_actions[:, :, 0]
        self.state['current_entities'] = self.current_entities
        return self.state


class env(object):
    def __init__(self, params, mode='train'):

        self.batch_size = params['batch_size']
        self.num_rollouts = params['num_rollouts']
        self.positive_reward = params['positive_reward']
        self.negative_reward = params['negative_reward']
        self.mode = mode
        self.path_len = params['path_length']
        self.test_rollouts = params['test_rollouts']
        self.embedding_dim = params['embedding_size']
        self.input_dir = params['data_input_dir']
        # print self.embedding_dim
        # print params['embedding_size']
        # print 'input_dir:', input_dir


        # add for GAIL
        ##########

        # f1 = open(input_dir + '/entity2id.txt')
        # f2 = open(input_dir + '/relation2id.txt')
        # self.entity2id = f1.readlines()
        # self.relation2id = f2.readlines()
        # f1.close()
        # f2.close()
        # self.entity2id_ = {}
        # self.relation2id_ = {}
        # self.relations = []
        # for line in self.entity2id:
        #     # dict{entity:id}
        #     self.entity2id_[line.split()[0]] = int(line.split()[1])
        # for line in self.relation2id:
        #     # dict{relation:id}
        #     self.relation2id_[line.split()[0]] = int(line.split()[1])
        #     # list[all relations]
        #     self.relations.append(line.split()[0])
        # self.entity2vec = np.loadtxt(input_dir + '/entity2vec.bern')  # TransR
        self.entity2id_ = params['entity_vocab']
        self.relation2id_ = params['relation_vocab']
        self.relation2vec = np.loadtxt(self.input_dir + '/relation2vec.bern')  # TransR
        # print self.relation2vec.shape
        if len(self.relation2id_) > len(self.relation2vec):
            diff = len(self.relation2id_) - len(self.relation2vec)
            diff_padding = np.random.random((diff,self.relation2vec.shape[1]))
            self.relation2vec = np.concatenate((self.relation2vec, diff_padding), axis=0)


        # print diff_padding.shape
        # print self.relation2vec.shape
        # print self.relation2id_['PAD']
        # print self.relation2id_ == params['relation_vocab']

        # raw_input('----------')

        # self.path = []
        # self.path_relations = []
        # KG used to build the RL environment for path finding
        # raw.kb + inv_relation addition => kb_env_rl
        # so the scale of kb_env_rl is double than raw.kb
        f = open(self.input_dir + '/kb_env_rl.txt')
        kb_all = f.readlines()
        f.close()
        self.kb = []
        # print 'input_dir:', input_dir
        relation = self.input_dir.split('/')[-2]
        print relation
        if relation is not None:
            for line in kb_all:
                # index here decided by the format of data set
                rel = line.split()[2].split(':')[-1]
                if rel != relation and rel != relation + '_inv':
                    self.kb.append(line)
        print 'size of kb:', len(self.kb)
        ##########

        # loaded all data
        if mode == 'train':
            self.batcher = RelationEntityBatcher(input_dir=self.input_dir,
                                                 batch_size=params['batch_size'],
                                                 entity_vocab=params['entity_vocab'],
                                                 relation_vocab=params['relation_vocab']
                                                 )
        else:  # test
            self.batcher = RelationEntityBatcher(input_dir=self.input_dir,
                                                 mode =mode,
                                                 batch_size=params['batch_size'],
                                                 entity_vocab=params['entity_vocab'],
                                                 relation_vocab=params['relation_vocab'])

            self.total_no_examples = self.batcher.store.shape[0]
        self.grapher = RelationEntityGrapher(triple_store=params['data_input_dir'] + '/' + 'graph.txt',
                                             max_num_actions=params['max_num_actions'],
                                             entity_vocab=params['entity_vocab'],
                                             relation_vocab=params['relation_vocab'])

    def get_episodes(self):
        params = self.batch_size, self.path_len, self.num_rollouts, self.test_rollouts, self.positive_reward, self.negative_reward, self.mode, self.batcher
        if self.mode == 'train':
            for data in self.batcher.yield_next_batch_train():
                yield Episode(self.grapher, data, params)
        else:
            for data in self.batcher.yield_next_batch_test():
                if data == None:
                    return
                yield Episode(self.grapher, data, params)

    # add for GAIL
    ##########
    def get_valid_actions(self, entityID):
        actions = set()
        for line in self.kb:
            triple = line.split()
            e1_idx = self.entity2id_[triple[0]]
            if e1_idx == entityID:
                actions.add(self.relation2id_[triple[2]])
        return np.array(list(actions))

    def path_embedding(self, path):
        embeddings = [self.relation2vec[self.relation2id_[relation], :] for relation in path]
        # print self.relation2vec.shape
        # print len(embeddings[0])
        # print self.embedding_dim
        embeddings = np.reshape(embeddings, (-1, 2* self.embedding_dim))
        # simple sum? In the viewpoint of Trans, e_begin+r1+r2+...+rn=e_target => e_begin+path_embedding=e_target
        path_encoding = np.sum(embeddings, axis=0)
        return np.reshape(path_encoding, (-1, 2* self.embedding_dim))
