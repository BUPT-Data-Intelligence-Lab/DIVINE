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

os.environ["CUDA_VISIBLE_DEVICES"] = '0'  # use GPU with ID
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.8  # maximun alloc gpu80% of MEM
config.gpu_options.allow_growth = True  # allocate dynamically

relation = sys.argv[1]
# dataPath = '../NELL-995/' was settled in utils.py
graphpath = dataPath + 'tasks/' + relation + '/' + 'graph.txt'
relationPath = dataPath + 'tasks/' + relation + '/' + 'train_pos'
demoPath = dataPath + 'tasks/' + relation + '/' + 'demo_path.txt'
negPath = dataPath + 'tasks/' + relation + '/' + 'neg_path.txt'


def train_discriminator(real_inputs, random_inputs, fake_inputs_batch, discriminator, epochs):
    print '---------- train discriminator ----------'
    for ind in range(epochs):
        discriminator.update(real_inputs, fake_inputs_batch)
        if ind % (epochs / 10) == 0:
            disc_cost, gen_reward = discriminator.predict(real_inputs, fake_inputs_batch)
            if math.isnan(disc_cost):
                print('----- retry -----')
                return False

    print('----------real_inputs----------')
    disc_cost, gen_reward = discriminator.predict(real_inputs, real_inputs)
    print('disc_cost:', disc_cost)
    print('gen_reward:', gen_reward)

    print('----------random_inputs----------')
    disc_cost, gen_reward_random = discriminator.predict(real_inputs, random_inputs)
    print('disc_cost:', disc_cost)
    print('gen_reward:', gen_reward_random)

    print('----------fake_inputs----------')
    disc_cost, gen_reward_fake = discriminator.predict(real_inputs, fake_inputs_batch)
    print('disc_cost:', disc_cost)
    print('gen_reward:', gen_reward_fake)

    return gen_reward_random


def reinforcement(train_data, real_inputs_all, fake_inputs, batch_size, generator, discriminator):
    num_samples = len(train_data)
    batch_num = int(fake_inputs.shape[0] / batch_size)
    env = Env(dataPath)
    f = open(demoPath)
    demo_paths = np.array([item.strip().rsplit(' -> ') for item in f.readlines()])
    f.close()
    paths_first_id = [env.relation2id_[path[0]] for path in demo_paths]
    # real_input_shape = real_inputs_all[:batch_size]
    # random_inputs = np.random.random(real_inputs_all[:batch_size].shape)
    # print random_inputs.shape
    epochs = 1
    gen_reward_random = 0
    # print '---------- train discriminator ----------'
    for i in range(batch_num):
        fake_inputs_batch = fake_inputs[batch_size * i:batch_size * (i + 1)]
        for episode in xrange(num_samples):
            # print "Episode %d" % episode
            env = Env(dataPath, train_data[episode % num_samples])
            sample = train_data[episode % num_samples].split()
            print sample
            # print env.entity2id_[sample[0]]

            # according to the topology of KG to get dynamic demo paths
            valid_actions = env.get_valid_actions(env.entity2id_[sample[0]])
            valid_path_idx = [idx for idx, action_id in enumerate(paths_first_id) if action_id in valid_actions]
            valid_path_num = len(valid_path_idx)

            if valid_path_num == 0:
                real_inputs = real_inputs_all[:batch_size]
            elif valid_path_num >= batch_size:
                real_inputs_idx = valid_path_idx[:batch_size]
                real_inputs = real_inputs_all[real_inputs_idx]
            else:
                diff = batch_size - valid_path_num
                paths_context_idx = [idx for idx, action_id in enumerate(paths_first_id) if idx not in valid_path_idx]
                context_paths = real_inputs_all[paths_context_idx]
                query_paths = real_inputs_all[valid_path_idx]
                semantic_sim = np.sum(np.dot(np.reshape(context_paths,(-1,embedding_dim)),np.reshape(query_paths,(-1,embedding_dim)).T),axis=1)
                padding_paths_idx = np.array(paths_context_idx)[np.argsort(-semantic_sim)[:diff]]
                real_inputs_idx = valid_path_idx + padding_paths_idx.tolist()
                real_inputs =real_inputs_all[real_inputs_idx]

            random_inputs = np.random.random(real_inputs.shape)



            gen_reward_random = train_discriminator(real_inputs, random_inputs, fake_inputs_batch, discriminator, epochs)
            # If the training is unstable, retry
            retry_times = 0
            while (gen_reward_random is False) and (retry_times <= 5):
                retry_times += 1
                print('retry times:', retry_times)
                gen_reward_random = train_discriminator(real_inputs, random_inputs, fake_inputs_batch, discriminator,
                                                        epochs)

    print '---------- train generator ----------'

    # for train generator
    success = 0
    path_found = []
    path_found_total = []
    path_relation_found = []
    invalid_path_found = []
    state_negative_list = []
    action_negative_list = []
    transitions_list = []
    # num_samples = 50
    for episode in xrange(num_samples):
        print '\nTrain sample %d/%d: %s' % (episode, num_samples, train_data[episode][:-1])
        env = Env(dataPath, train_data[episode])
        sample = train_data[episode].split()
        state_idx = [env.entity2id_[sample[0]], env.entity2id_[sample[1]], 0]

        # according to the topology of KG to get dynamic demo paths
        valid_actions = env.get_valid_actions(env.entity2id_[sample[0]])
        valid_path_idx = [idx for idx, action_id in enumerate(paths_first_id) if action_id in valid_actions]
        if len(valid_path_idx) >= batch_size:
            real_inputs_idx = valid_path_idx[:batch_size]
            real_inputs = np.array([real_inputs_all[idx] for idx in real_inputs_idx])
        else:
            diff = batch_size - len(valid_path_idx)
            paths_extend_idx = [idx for idx, action_id in enumerate(paths_first_id) if idx not in valid_path_idx][
                               :diff]
            real_inputs_idx = valid_path_idx + paths_extend_idx
            real_inputs = np.array([real_inputs_all[idx] for idx in real_inputs_idx])
        random_inputs = np.random.random(real_inputs.shape)

        transitions = []
        state_batch_negative = []
        action_batch_negative = []

        for t in count():
            state_vec = env.idx_state(state_idx)
            action_probs = generator.predict(state_vec)
            action_probs = np.squeeze(action_probs)
            action_chosen = np.random.choice(np.arange(action_space), p=action_probs)
            reward, new_state, done = env.interact(state_idx, action_chosen)
            # print reward
            # this part is different with test generator
            if reward == -1:  # the action fails for this step
                state_batch_negative.append(state_vec)
                action_batch_negative.append(action_chosen)

            new_state_vec = env.idx_state(new_state)
            transitions.append(
                Transition(state=state_vec, action=action_chosen, next_state=new_state_vec, reward=reward))
            if done or t == max_steps:
                if done:
                    success += 1
                    print "Success"
                    path = path_clean(' -> '.join(env.path))
                    path_found.append(path)
                    path_found_total.append(path)
                    transitions_list.append(transitions)
                else:
                    print 'Episode ends due to step limit'
                    invalid_path = path_clean(' -> '.join(env.path))
                    invalid_path_found.append(invalid_path)
                    state_negative_list.append(state_batch_negative)
                    action_negative_list.append(action_batch_negative)
                    print 'Do one teacher guideline'
                    good_episodes, path_str = teacher(sample[0], sample[1], env, graphpath, num_paths=1)
                    if good_episodes is False:
                        print 'No find good_episodes'
                    else:
                        for item in good_episodes:
                            teacher_state_batch = []
                            teacher_action_batch = []
                            for _, transition in enumerate(item):
                                teacher_state_batch.append(transition.state)
                                teacher_action_batch.append(transition.action)
                            teacher_state_batch = np.squeeze(teacher_state_batch)
                            teacher_state_batch = np.reshape(teacher_state_batch, [-1, state_dim])
                            generator.update(teacher_state_batch, 1, teacher_action_batch)

                break
            state_idx = new_state

        print 'path_found_total:', len(path_found_total)
        print 'invalid_path_found:', len(invalid_path_found)

        # Update the agent when it found valid paths
        if len(path_found) % batch_size == 0 and len(path_found) > 0:
            path_found_batch = []
            for path in path_found[-batch_size:]:
                rel_ent = path.split(' -> ')
                path_relation = []
                for idx, item in enumerate(rel_ent):
                    if idx % 2 == 0:
                        path_relation.append(item)
                path_found_batch.append(' -> '.join(path_relation))
            path_found_inputs = np.array([env.path_embedding(item.strip().rsplit(' -> ')) for item in path_found_batch])
            path_found_inputs = path_found_inputs.astype('float32')
            # print path_found_inputs.shape
            _, gen_reward = discriminator.predict(real_inputs, path_found_inputs)
            gen_reward_found = (gen_reward - gen_reward_random) / batch_size
            print 'update generator by gen_reward_found:', gen_reward_found
            print len(transitions_list)
            for i in range(batch_size):
                state_found = []
                action_found = []
                for t, transition in enumerate(transitions_list[i]):
                    if transition.reward == 0:
                        state_found.append(transition.state)
                        action_found.append(transition.action)
                generator.update(np.reshape(state_found, (-1, state_dim)), max(gen_reward_found, 0), action_found)
            # clear batch cache of found paths
            transitions_list = []
            path_found = []
            # update discriminator
            gen_reward_random = train_discriminator(real_inputs, random_inputs, path_found_inputs, discriminator,
                                                    epochs=batch_size)
            # If the training is unstable, retry
            retry_times = 0
            while (gen_reward_random is False) and (retry_times <= 5):
                retry_times += 1
                print('retry times:', retry_times)
                gen_reward_random = train_discriminator(real_inputs, random_inputs, path_found_inputs, discriminator,
                                                        epochs=batch_size)

        # Discourage the agent when it choose invalid paths
        if len(invalid_path_found) % batch_size == 0 and len(invalid_path_found) > 0:
            try:
                invalid_batch = []
                for path in invalid_path_found[-batch_size:]:
                    rel_ent = path.split(' -> ')
                    path_relation = []
                    for idx, item in enumerate(rel_ent):
                        if idx % 2 == 0:
                            path_relation.append(item)
                    invalid_batch.append(' -> '.join(path_relation))
                invalid_inputs = np.array([env.path_embedding(item.strip().rsplit(' -> ')) for item in invalid_batch])
                invalid_inputs = invalid_inputs.astype('float32')
                # print invalid_inputs.shape
                _, gen_reward = discriminator.predict(real_inputs, invalid_inputs)
                gen_reward_invalid = (gen_reward_random - gen_reward) / batch_size
                print 'Penalty to invalid steps by gen_reward_invalid:', gen_reward_invalid
                # print len(state_negative_list)
                # print len(action_negative_list)
                for i in range(batch_size):
                    state_negative = np.reshape(state_negative_list[i], (-1, state_dim))
                    action_negative = action_negative_list[i]
                    generator.update(state_negative, min(gen_reward_invalid, 0), action_negative)
            except Exception as e:
                print 'Penalty failed'
                continue
            # clear batch cache of invalid paths
            state_negative_list = []
            action_negative_list = []
            invalid_path_found = []
            # update discriminator
            gen_reward_random = train_discriminator(real_inputs, random_inputs, invalid_inputs, discriminator,
                                                    epochs=batch_size)
            # If the training is unstable, retry
            retry_times = 0
            while (gen_reward_random is False) and (retry_times <= 5):
                retry_times += 1
                print('retry times:', retry_times)
                gen_reward_random = train_discriminator(real_inputs, random_inputs, invalid_inputs, discriminator,
                                                        epochs=batch_size)
    print 'Success percentage:', success / num_samples

    for path in path_found_total:
        rel_ent = path.split(' -> ')
        path_relation = []
        for idx, item in enumerate(rel_ent):
            if idx % 2 == 0:
                path_relation.append(item)
        path_relation_found.append(' -> '.join(path_relation))

    relation_path_stats = collections.Counter(path_relation_found).items()
    relation_path_stats = sorted(relation_path_stats, key=lambda x: x[1], reverse=True)

    f = open(dataPath + 'tasks/' + relation + '/' + 'path_stats_test.txt', 'w')
    for item in relation_path_stats:
        f.write(item[0] + '\t' + str(item[1]) + '\n')
    f.close()
    print 'Path stats saved'


def retrain():
    tf.reset_default_graph()
    env = Env(dataPath)

    f = open(demoPath)
    real_inputs_all = np.array([env.path_embedding(item.strip().rsplit(' -> ')) for item in f.readlines()])
    real_inputs_all = real_inputs_all.astype('float32')  # (paths_number,1,embedding_dim)
    f.close()



    f = open(negPath)
    fake_inputs = np.array([env.path_embedding(item.strip().rsplit(' -> ')) for item in f.readlines()])
    fake_inputs = fake_inputs.astype('float32')
    f.close()

    batch_size = 5
    # real_inputs = real_inputs[:batch_size]
    # fake_inputs = fake_inputs
    # random_inputs = np.random.random(real_inputs.shape)
    print real_inputs_all.shape
    # print random_inputs.shape
    print fake_inputs.shape

    f = open(relationPath)
    train_data = f.readlines()  # positive sample(h,t,r) from this task in KG
    f.close()

    discriminator = Discriminator(batch_size)
    generator = Generator()
    saver = tf.train.Saver()
    with tf.Session(config=config) as sess:
        saver.restore(sess, 'models/pre_train_model_' + relation)
        print 'pre-trained model restored'
        reinforcement(train_data, real_inputs_all, fake_inputs, batch_size, generator, discriminator)
        saver.save(sess, 'models/retrained_model_' + relation)
        print 'retrained model saved'


def test():  # same as test generator in pretrain
    tf.reset_default_graph()
    generator = Generator()

    # relationPath = dataPath + 'tasks/' + relation + '/' + 'train_pos'
    # test_data = train data = KG positive samples
    # test_data here are only used to test the result of retrain().
    # The whole model test data are from dataPath_ + '/sort_test.pairs'
    f = open(relationPath)
    all_data = f.readlines()
    f.close()

    test_data = all_data
    test_num = len(test_data)

    success = 0

    saver = tf.train.Saver()
    path_found = []
    path_relation_found = []
    # path_set = set()

    with tf.Session(config=config) as sess:
        saver.restore(sess, 'models/retrained_model_' + relation)
        print 'retrained model reloaded'

        for episode in xrange(test_num):
            print 'Test sample %d/%d: %s' % (episode, test_num, test_data[episode][:-1])
            env = Env(dataPath, test_data[episode])
            sample = test_data[episode].split()
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

                if done or t == max_steps_test:
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
    print 'Success persentage:', success / test_num

    f = open(dataPath + 'tasks/' + relation + '/' + 'path_to_use_test.txt', 'w')
    for item in ranking_path:
        f.write(item[0] + '\n')
    f.close()
    print 'path to use saved'
    return


def merge_path():
    f = open(demoPath)
    demo_paths = [item.strip() for item in f.readlines()]
    f.close()
    # print 'demo paths:\n', demo_paths

    with open(dataPath + 'tasks/' + relation + '/' + 'path_to_use_test.txt') as f:
        test_paths = [item.strip() for item in f.readlines()]

    # print 'test paths:\n', test_paths

    with open(dataPath + 'tasks/' + relation + '/' + 'path_to_use_total.txt', 'w') as f:
        for path in demo_paths:
            f.write(path + '\n')
        for path in test_paths:
            if path not in demo_paths:
                f.write(path + '\n')
    print 'path_to_use_total saved'


if __name__ == '__main__':
    retrain()
    test()
    merge_path()
