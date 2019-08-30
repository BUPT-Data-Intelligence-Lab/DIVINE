# -*- coding: utf-8 -*-
from __future__ import division
import tensorflow as tf
import numpy as np
import collections
import math
import sys
from itertools import count
from utils import *
from env import Env
from BFS.KB import KB
from BFS.BFS import BFS
import time
import os

from generator import *
from discriminator import *

os.environ["CUDA_VISIBLE_DEVICES"] = '1' #use GPU with ID
config = tf.ConfigProto()  
config.gpu_options.per_process_gpu_memory_fraction = 0.8 # maximun alloc gpu80% of MEM  
config.gpu_options.allow_growth = True #allocate dynamically  

relation = sys.argv[1]
# dataPath = '../NELL-995/' was settled in utils.py
graphpath = dataPath + 'tasks/' + relation + '/' + 'graph.txt'
relationPath = dataPath + 'tasks/' + relation + '/' + 'train_pos'
demoPath = dataPath + 'tasks/' + relation + '/' + 'demo_path.txt'


def negative_sampling():  # get neg
    tf.reset_default_graph()
    generator = Generator()

    f = open(relationPath)
    train_data = f.readlines()  # positive sample(h,t,r) from this task in KG
    f.close()

    num_samples = len(train_data)

    # for testing generator
    success = 0
    path_found = []
    path_relation_found = []

    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())

        for episode in xrange(num_samples):
            print "Episode %d" % episode
            # print 'Training Sample:', train_data[episode % num_samples][:-1]  # del the '\n' in the last position
            env = Env(dataPath, train_data[episode % num_samples])
            sample = train_data[episode % num_samples].split()

            # good_episodes = [[Transition(state, action_id, next_state, reward)], ...]
            # state.shape = (1, 200) type(state) = <type 'numpy.ndarray'>
            good_episodes, path_str = teacher(sample[0], sample[1], env, graphpath, random_mechanism=True)

            if good_episodes is False:
                print 'No find good_episodes'
            else:
                for item in good_episodes:
                    state_batch = []
                    action_batch = []
                    for t, transition in enumerate(item):
                        state_batch.append(transition.state)
                        action_batch.append(transition.action)
                    # Remove single-dimensional entries from the shape of an array.
                    state_batch = np.squeeze(state_batch)
                    state_batch = np.reshape(state_batch, [-1, state_dim])
                    generator.update(state_batch, 1, action_batch)  # target=1

        print '---------- test generator ----------'

        for episode in xrange(num_samples):
            print 'Test sample %d: %s' % (episode, train_data[episode][:-1])
            env = Env(dataPath, train_data[episode])
            sample = train_data[episode].split()
            state_idx = [env.entity2id_[sample[0]], env.entity2id_[sample[1]], 0]

            transitions = []

            for t in count():
                state_vec = env.idx_state(state_idx)
                action_probs = generator.predict(state_vec)
                action_probs = np.squeeze(action_probs)
                action_chosen = np.random.choice(np.arange(action_space), p=action_probs)
                reward, new_state, done = env.interact(state_idx, action_chosen)
                new_state_vec = env.idx_state(new_state)
                transitions.append(
                    Transition(state=state_vec, action=action_chosen, next_state=new_state_vec, reward=reward))

                if done or t == max_steps_test:  # max_steps_test = 50
                    if done:
                        success += 1
                        print "Success\n"
                        path = path_clean(' -> '.join(env.path))
                        path_found.append(path)
                    else:
                        print 'Episode ends due to step limit\n'
                    break
                state_idx = new_state

    for path in path_found:
        rel_ent = path.split(' -> ')
        path_relation = []
        for idx, item in enumerate(rel_ent):
            if idx % 2 == 0:
                path_relation.append(item)
        path_relation_found.append(' -> '.join(path_relation))

    # path_stats = collections.Counter(path_found).items()
    relation_path_stats = collections.Counter(path_relation_found).items()
    # we prefer the path found many times
    relation_path_stats = sorted(relation_path_stats, key=lambda x: x[1], reverse=True)

    ranking_path = []
    for item in relation_path_stats:
        path = item[0]
        length = len(path.split(' -> '))
        ranking_path.append((path, length))

    # we prefer short path
    ranking_path = sorted(ranking_path, key=lambda x: x[1])
    print 'Success persentage:', success / num_samples

    f = open(dataPath + 'tasks/' + relation + '/' + 'neg_path.txt', 'w')
    for item in ranking_path:
        f.write(item[0] + '\n')
    f.close()
    print 'negative path saved'
    return


def pre_train():  # iterations to train for in WGAN-GP is 200000
    tf.reset_default_graph()
    env = Env(dataPath)

    f = open(demoPath)
    real_inputs_all = np.array([env.path_embedding(item.strip().rsplit(' -> ')) for item in f.readlines()])
    real_inputs_all = real_inputs_all.astype('float32')  # (paths_number,1,embedding_dim)
    f.close()

    f = open(demoPath)
    demo_paths = np.array([item.strip().rsplit(' -> ') for item in f.readlines()])
    f.close()
    # print demo_paths

    # demo paths transfer to id
    # paths_id = [[env.relation2id_[rel] for rel in path] for path in demo_paths]
    paths_first_id = [env.relation2id_[path[0]] for path in demo_paths]
    # print paths_id
    # print paths_first_id


    # batch_size = len(real_inputs)

    print real_inputs_all.shape
    batch_size = 5

    f = open(relationPath)
    train_data = f.readlines()  # positive sample(h,t,r) from this task in KG
    f.close()
    num_samples = len(train_data)

    topo_stats = dict()
    with open(dataPath + 'tasks/' + relation + '/' + 'padding_log.txt', 'w') as f:
        f.write('num_samples' + '\t' + str(num_samples) + '\n')
    print 'reset padding log'

    discriminator = Discriminator(batch_size)
    generator = Generator()
    saver = tf.train.Saver()
    with tf.Session(config = config) as sess:
        sess.run(tf.global_variables_initializer())

        print '---------- pre-train discriminator ----------'

        for episode in xrange(num_samples):
            print "Episode %d" % episode
            env = Env(dataPath, train_data[episode % num_samples])
            sample = train_data[episode % num_samples].split()
            print sample
            # print env.entity2id_[sample[0]]

            # according to the topology of KG to get dynamic demo paths
            valid_actions = env.get_valid_actions(env.entity2id_[sample[0]])
            # print 'valid actions:',valid_actions

            topo_degree = len(valid_actions)
            if topo_degree in topo_stats:
                topo_stats[topo_degree] += 1
            else:
                topo_stats[topo_degree] = 1

            valid_path_idx = [idx for idx, action_id in enumerate(paths_first_id) if action_id in valid_actions]
            # print 'valid_path_idx:',valid_path_idx
            valid_path_num = len(valid_path_idx)

            if valid_path_num == 0:
                diff = batch_size - valid_path_num
                real_inputs = real_inputs_all[:batch_size]
                with open(dataPath + 'tasks/' + relation + '/' + 'padding_log.txt', 'a') as f:
                    f.write('\t'.join(sample) + '\t' + str(diff) + '\n')
            elif valid_path_num >= batch_size:
                real_inputs_idx = valid_path_idx[:batch_size]
                # print 'real_inputs_idx:',real_inputs_idx
                real_inputs = real_inputs_all[real_inputs_idx]
                # print real_inputs
                # print real_inputs.shape
            else:
                diff = batch_size - valid_path_num
                paths_context_idx = [idx for idx, action_id in enumerate(paths_first_id) if idx not in valid_path_idx]
                print 'paths_context_idx:',paths_context_idx
                print 'valid_path_idx:',valid_path_idx
                context_paths = real_inputs_all[paths_context_idx]
                query_paths = real_inputs_all[valid_path_idx]
                print context_paths.shape
                print query_paths.shape
                semantic_sim = np.sum(np.dot(np.reshape(context_paths,(-1,embedding_dim)),np.reshape(query_paths,(-1,embedding_dim)).T),axis=1)
                # print semantic_sim
                padding_paths_idx = np.array(paths_context_idx)[np.argsort(-semantic_sim)[:diff]]
                print padding_paths_idx
                # paths_extend_idx = [idx for idx, action_id in enumerate(paths_first_id) if idx not in valid_path_idx][
                #                    :diff]
                # print 'paths_extend_idx:',paths_extend_idx
                real_inputs_idx = valid_path_idx + padding_paths_idx.tolist()
                print 'real_inputs_idx:',real_inputs_idx
                real_inputs =real_inputs_all[real_inputs_idx]
                # print real_inputs
                print real_inputs.shape
                with open(dataPath + 'tasks/' + relation + '/' + 'padding_log.txt', 'a') as f:
                    f.write('\t'.join(sample) + '\t' + str(diff) + '\n')
                print 'padding log updated'

            random_inputs = np.random.random(real_inputs.shape)

            discriminator.update(real_inputs, random_inputs)
            if episode % (num_samples / 10) == 0:
                disc_cost, gen_reward = discriminator.predict(real_inputs, random_inputs)
                if math.isnan(disc_cost):
                    print('----- retry -----')
                    return False
                print('----------')
                print('disc_cost:', disc_cost)
                print('gen_reward:', gen_reward)

        topo_stats_list = sorted(topo_stats.items(), key=lambda x: x[0])
        with open(dataPath + 'tasks/' + relation + '/' + 'topo_stats.txt', 'w') as f:
            for item in topo_stats_list:
                f.write(str(item[0]) + '\t' + str(item[1]) + '\n')
        print 'topo stats saved'


        print '---------- pre train generator ----------'

        for episode in xrange(num_samples):
            print "Episode %d" % episode
            # print 'Training Sample:', train_data[episode % num_samples][:-1]  # del the '\n' in the last position

            env = Env(dataPath, train_data[episode % num_samples])
            sample = train_data[episode % num_samples].split()

            # good_episodes = [[Transition(state, action_id, next_state, reward)], ...]
            # state.shape = (1, 200) type(state) = <type 'numpy.ndarray'>
            good_episodes, path_str = teacher(sample[0], sample[1], env, graphpath)

            if good_episodes is False:
                print 'No find good_episodes'
            else:
                for item in good_episodes:
                    state_batch = []
                    action_batch = []
                    for t, transition in enumerate(item):
                        state_batch.append(transition.state)
                        action_batch.append(transition.action)
                    # Remove single-dimensional entries from the shape of an array.
                    state_batch = np.squeeze(state_batch)
                    state_batch = np.reshape(state_batch, [-1, state_dim])
                    generator.update(state_batch, 1, action_batch)  # target=1

        saver.save(sess, 'models/pre_train_model_' + relation)
        print 'pre-trained model saved'
    return gen_reward



if __name__ == "__main__":
    # negative_sampling()
    suc = pre_train()
    retry_times = 0
    while (suc is False) and (retry_times <= 5):
        # print('suc:',suc)
        retry_times += 1
        print('retry times:', retry_times)
        suc = pre_train()
