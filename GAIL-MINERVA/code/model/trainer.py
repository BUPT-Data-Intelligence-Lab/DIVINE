from __future__ import absolute_import
from __future__ import division
from tqdm import tqdm
import json
import time
import os
import logging
import numpy as np
import tensorflow as tf
from code.model.agent import Agent
from code.options import read_options
from code.model.environment import env
import codecs
from collections import defaultdict
import gc
import resource
import sys
from code.model.baseline import ReactiveBaseline
from code.model.nell_eval import nell_eval
from scipy.misc import logsumexp as lse
from discriminator import *

logger = logging.getLogger()
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)


class Trainer(object):
    def __init__(self, params):

        # transfer parameters to self
        for key, val in params.items(): setattr(self, key, val);

        self.agent = Agent(params)
        self.save_path = None
        self.train_environment = env(params, 'train')  # loaded train data here
        self.dev_test_environment = env(params, 'dev')  # loaded dev data here
        self.test_test_environment = env(params, 'test')  # loaded test data here
        self.test_environment = self.dev_test_environment
        self.rev_relation_vocab = self.train_environment.grapher.rev_relation_vocab
        self.rev_entity_vocab = self.train_environment.grapher.rev_entity_vocab
        self.max_hits_at_10 = 0
        self.disc_size = 5
        self.ePAD = self.entity_vocab['PAD']
        self.rPAD = self.relation_vocab['PAD']
        # optimize
        self.baseline = ReactiveBaseline(l=self.Lambda)
        self.optimizer = tf.train.AdamOptimizer(self.learning_rate)
        self.input_dir = params['data_input_dir']
        self.disc_embedding_size = 2 * params['embedding_size']
        self.discriminator = Discriminator(self.disc_size, self.disc_embedding_size)
        self.num_rollouts = params['num_rollouts']
        self.num_iter = params['total_iterations']

    def calc_reinforce_loss(self): # only used in initialize() once
        loss = tf.stack(self.per_example_loss, axis=1)  # [B, T]

        self.tf_baseline = self.baseline.get_baseline_value()
        # multiply with rewards
        final_reward = self.final_reward - self.tf_baseline
        reward_mean, reward_var = tf.nn.moments(final_reward, axes=[0, 1])
        # Constant added for numerical stability
        reward_std = tf.sqrt(reward_var) + 1e-6
        final_reward = tf.div(final_reward - reward_mean, reward_std)

        loss = tf.multiply(loss, final_reward)  # [B, T]
        self.loss_before_reg = loss

        total_loss = tf.reduce_mean(loss) - self.decaying_beta * self.entropy_reg_loss(self.per_example_logits)  # scalar

        return total_loss

    def entropy_reg_loss(self, all_logits):
        all_logits = tf.stack(all_logits, axis=2)  # [B, MAX_NUM_ACTIONS, T]
        entropy_policy = - tf.reduce_mean(tf.reduce_sum(tf.multiply(tf.exp(all_logits), all_logits), axis=1))  # scalar
        return entropy_policy

    def initialize(self, restore=None, sess=None):

        logger.info("Creating TF graph...")
        self.candidate_relation_sequence = []
        self.candidate_entity_sequence = []
        self.input_path = []
        self.first_state_of_test = tf.placeholder(tf.bool, name="is_first_state_of_test")
        self.query_relation = tf.placeholder(tf.int32, [None], name="query_relation")
        self.range_arr = tf.placeholder(tf.int32, shape=[None, ])
        self.global_step = tf.Variable(0, trainable=False)
        self.decaying_beta = tf.train.exponential_decay(self.beta, self.global_step, 200, 0.90, staircase=False)
        self.entity_sequence = []
        self.final_reward = tf.placeholder(tf.float32, [None, 1], name="final_reward")

        for t in range(self.path_length):
            next_possible_relations = tf.placeholder(tf.int32, [None, self.max_num_actions], name="next_relations_{}".format(t))
            next_possible_entities = tf.placeholder(tf.int32, [None, self.max_num_actions], name="next_entities_{}".format(t))
            input_label_relation = tf.placeholder(tf.int32, [None], name="input_label_relation_{}".format(t))
            start_entities = tf.placeholder(tf.int32, [None, ])
            self.input_path.append(input_label_relation)
            self.candidate_relation_sequence.append(next_possible_relations)
            self.candidate_entity_sequence.append(next_possible_entities)
            self.entity_sequence.append(start_entities)

        self.loss_before_reg = tf.constant(0.0)

        self.per_example_loss, self.per_example_logits, self.action_idx = self.agent(
            self.candidate_relation_sequence,
            self.candidate_entity_sequence, self.entity_sequence,
            self.input_path,
            self.query_relation, self.range_arr, self.first_state_of_test, self.path_length)


        # calculate reinfor
        self.loss_op = self.calc_reinforce_loss()

        # backprop
        # self.bp and self.train_op only used once here
        self.train_op = self.bp(self.loss_op)

        # Building the test graph
        self.prev_state = tf.placeholder(tf.float32, self.agent.get_mem_shape(), name="memory_of_agent")
        self.prev_relation = tf.placeholder(tf.int32, [None, ], name="previous_relation")
        self.query_embedding = tf.nn.embedding_lookup(self.agent.relation_lookup_table, self.query_relation)  # [B, 2D]
        layer_state = tf.unstack(self.prev_state, self.LSTM_layers)
        formated_state = [tf.unstack(s, 2) for s in layer_state]
        self.next_relations = tf.placeholder(tf.int32, shape=[None, self.max_num_actions])
        self.next_entities = tf.placeholder(tf.int32, shape=[None, self.max_num_actions])

        self.current_entities = tf.placeholder(tf.int32, shape=[None,])

        with tf.variable_scope("policy_steps_unroll") as scope:
            scope.reuse_variables()
            self.test_loss, test_state, self.test_logits, self.test_action_idx, self.chosen_relation = self.agent.step(
                self.next_relations, self.next_entities, formated_state, self.prev_relation, self.query_embedding,
                self.current_entities, self.input_path[0], self.range_arr, self.first_state_of_test)
            self.test_state = tf.stack(test_state)

        logger.info('TF Graph creation done..')
        self.model_saver = tf.train.Saver(max_to_keep=5)

        # return the variable initializer Op.
        if not restore:
            return tf.global_variables_initializer()
        else:
            return self.model_saver.restore(sess, restore)

    def initialize_pretrained_embeddings(self, sess):
        if self.pretrained_embeddings_action != '':  # but the default value is ''
            embeddings = np.loadtxt(open(self.pretrained_embeddings_action))
            _ = sess.run((self.agent.relation_embedding_init),
                         feed_dict={self.agent.action_embedding_placeholder: embeddings})
        if self.pretrained_embeddings_entity != '':
            embeddings = np.loadtxt(open(self.pretrained_embeddings_entity))
            _ = sess.run((self.agent.entity_embedding_init),
                         feed_dict={self.agent.entity_embedding_placeholder: embeddings})

    def bp(self, cost):  # only used in initialize() once
        self.baseline.update(tf.reduce_mean(self.final_reward))
        tvars = tf.trainable_variables()
        grads = tf.gradients(cost, tvars)
        grads, _ = tf.clip_by_global_norm(grads, self.grad_clip_norm)
        train_op = self.optimizer.apply_gradients(zip(grads, tvars))
        with tf.control_dependencies([train_op]):  # see https://github.com/tensorflow/tensorflow/issues/1899
            self.dummy = tf.constant(0)
        return train_op

    def reshape_reward(self, rewards):  # only used in train() once
        reshaped_reward = np.zeros([rewards.shape[0], 1])  # [B, T]
        reshaped_reward[:, 0] = rewards
        return reshaped_reward

    def gpu_io_setup(self):  # only used in train() once
        # create fetches for partial_run_setup
        fetches = self.per_example_loss + self.action_idx + [self.loss_op] + self.per_example_logits + [self.dummy]
        feeds =  [self.first_state_of_test] + self.candidate_relation_sequence+ self.candidate_entity_sequence + self.input_path + \
                [self.query_relation] + [self.final_reward] + [self.range_arr] + self.entity_sequence

        feed_dict = [{} for _ in range(self.path_length)]

        feed_dict[0][self.first_state_of_test] = False
        feed_dict[0][self.query_relation] = None
        feed_dict[0][self.range_arr] = np.arange(self.batch_size*self.num_rollouts)
        for i in range(self.path_length):
            feed_dict[i][self.input_path[i]] = np.zeros(self.batch_size * self.num_rollouts)  # placebo
            feed_dict[i][self.candidate_relation_sequence[i]] = None
            feed_dict[i][self.candidate_entity_sequence[i]] = None
            feed_dict[i][self.entity_sequence[i]] = None

        return fetches, feeds, feed_dict

    def pre_train(self, sess):
        # tf.reset_default_graph()

        with open(self.input_dir + 'demo_path.txt') as f:
            real_inputs_all = np.array([self.train_environment.path_embedding(item.strip().rsplit(' -> ')) for item in f.readlines()])
            real_inputs_all = real_inputs_all.astype('float32')  # (paths_number,1,self.disc_embedding_size)

        with open(self.input_dir + 'demo_path.txt') as f:
            demo_paths = np.array([item.strip().rsplit(' -> ') for item in f.readlines()])
        # print demo_paths
        paths_first_id = [self.train_environment.relation2id_[path[0]] for path in demo_paths]
        print real_inputs_all.shape

        with open(self.input_dir + 'train.txt') as f:
            train_data = f.readlines()  # positive sample(h,t,r) from this task in KG

        num_samples = len(train_data)
        gen_reward = 0

        if num_samples < self.num_iter:
            num_epochs = self.num_iter
        else:
            num_epochs = num_samples

        # saver = tf.train.Saver()
        # with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())

        print '---------- pre-train discriminator ----------'

        for episode in xrange(num_epochs):
            print "Episode %d" % episode
            sample = train_data[episode % num_samples].split()
            # print sample
            # print env.entity2id_[sample[0]]

            # according to the topology of KG to get dynamic demo paths
            valid_actions = self.train_environment.get_valid_actions(self.train_environment.entity2id_[sample[0]])
            # print 'valid actions:',valid_actions

            valid_path_idx = [idx for idx, action_id in enumerate(paths_first_id) if action_id in valid_actions]
            # print 'valid_path_idx:',valid_path_idx
            valid_path_num = len(valid_path_idx)

            if valid_path_num == 0:
                real_inputs = real_inputs_all[:self.disc_size]
                # print 'real_inputs:', real_inputs.shape
            elif valid_path_num >= self.disc_size:
                real_inputs_idx = valid_path_idx[:self.disc_size]
                # print 'real_inputs_idx:',real_inputs_idx
                real_inputs = real_inputs_all[real_inputs_idx]
                # print real_inputs
                # print 'real_inputs:', real_inputs.shape
            else:
                diff = self.disc_size - valid_path_num
                paths_context_idx = [idx for idx, action_id in enumerate(paths_first_id) if
                                     idx not in valid_path_idx]
                # print 'paths_context_idx:', paths_context_idx
                # print 'valid_path_idx:', valid_path_idx
                context_paths = real_inputs_all[paths_context_idx]
                query_paths = real_inputs_all[valid_path_idx]
                # print context_paths.shape
                # print query_paths.shape
                semantic_sim = np.sum(np.dot(np.reshape(context_paths, (-1, self.disc_embedding_size)), np.reshape(query_paths, (-1, self.disc_embedding_size)).T), axis=1)
                # print semantic_sim
                padding_paths_idx = np.array(paths_context_idx)[np.argsort(-semantic_sim)[:diff]]
                # print 'padding_paths_idx:',padding_paths_idx
                # print 'paths_extend_idx:',paths_extend_idx
                real_inputs_idx = valid_path_idx + padding_paths_idx.tolist()
                # print 'real_inputs_idx:', real_inputs_idx
                real_inputs = real_inputs_all[real_inputs_idx]
                # print real_inputs
                # print 'real_inputs:',real_inputs.shape

            for _ in range(self.num_rollouts):
                random_inputs = np.random.random(real_inputs.shape)
                self.discriminator.update(real_inputs, random_inputs)

            if episode % (num_epochs // 100) == 0:
                disc_cost, gen_reward = self.discriminator.predict(real_inputs, random_inputs)
                if math.isnan(disc_cost):
                    print('----- retry -----')
                    return False
                print('----------')
                print('disc_cost:', disc_cost)
                print('gen_reward:', gen_reward)

        # saver.save(sess, input_path + 'models/pre_train_model_')
        # print 'pre-trained model saved'
        return gen_reward

    def train(self, sess):
        # import pdb
        # pdb.set_trace()
        fetches, feeds, feed_dict = self.gpu_io_setup()
        # print self.input_dir

        with open(self.input_dir + 'demo_path.txt') as f:
            real_inputs_all = np.array([self.train_environment.path_embedding(item.strip().rsplit(' -> ')) for item in f.readlines()])
            real_inputs_all = real_inputs_all.astype('float32')  # (paths_number,1,disc_embedding_dim)

        with open(self.input_dir + 'demo_path.txt') as f:
            demo_paths = np.array([item.strip().rsplit(' -> ') for item in f.readlines()])
        # print demo_paths
        paths_first_id = [self.train_environment.relation2id_[path[0]] for path in demo_paths]

        train_loss = 0.0
        self.batch_counter = 0
        # get_episodes() returns state conclude next_relations,next_entities and current_entities
        for episode in self.train_environment.get_episodes():

            self.batch_counter += 1
            h = sess.partial_run_setup(fetches=fetches, feeds=feeds)
            feed_dict[0][self.query_relation] = episode.get_query_relation()

            # get initial state
            state = episode.get_state()
            # for each time step
            # loss_before_regularization = []
            # logits = []
            candidate_paths = []
            for i in range(self.path_length):  # default path length is 3
                feed_dict[i][self.candidate_relation_sequence[i]] = state['next_relations']
                feed_dict[i][self.candidate_entity_sequence[i]] = state['next_entities']
                feed_dict[i][self.entity_sequence[i]] = state['current_entities']
                per_example_loss, per_example_logits, idx = sess.partial_run(h, [self.per_example_loss[i], self.per_example_logits[i], self.action_idx[i]], feed_dict=feed_dict[i])

                # print per_example_loss
                # print per_example_logits.shape
                # print 'idx:', len(idx)
                # print 'idx:',idx
                # print 'current_entities:', state['current_entities']
                # print 'next_relations:', state['next_relations'][::20][0][idx]
                # print 'next_entities:', state['next_entities'][::20][0]
                # raw_input('----------')

                candidate_paths.append(state['next_relations'][::20][0][idx])
                state = episode(idx)
                # print state['current_entities'][::20]
                # print state['next_relations'][::20]
                # print state['next_entities'][::20]

            # loss_before_regularization = np.stack(loss_before_regularization, axis=1)

            # print episode.current_entities
            # print episode.end_entities
            # get the final reward from the environment
            # if current_entitiy == end_entity reward = 1 else 0
            rewards = episode.get_reward()  # [batch_size*rollouts]
            valid_paths_idx = np.nonzero(rewards)[0]
            # print rewards
            # print valid_paths_idx
            candidate_paths = np.array(candidate_paths)
            # print candidate_paths.shape
            # print candidate_paths
            if len(valid_paths_idx) > 0:

                # get embedding of valid paths
                ##########
                valid_paths = candidate_paths.T[valid_paths_idx]
                valid_paths = np.array(list(set([tuple(t) for t in valid_paths])))
                # print 'disc_size:',self.disc_size
                # print 'valid_paths:',valid_paths
                # padding or slicing
                # print 'PAD id:', self.rPAD
                valid_nums = len(valid_paths)
                # print 'valid_nums:',valid_nums
                if valid_nums < self.disc_size:
                    valid_paths = np.pad(valid_paths,((0,self.disc_size - valid_nums),(0, 0)),'constant', constant_values=(self.rPAD,self.rPAD))
                else:
                    valid_paths = valid_paths[:self.disc_size]
                # print valid_paths
                # valid_paths_list = valid_paths.tolist()
                # print tf.nn.embedding_lookup(self.agent.relation_lookup_table, self.rPAD).eval()
                self.train_environment.relation2vec[self.rPAD] = tf.nn.embedding_lookup(self.agent.relation_lookup_table, self.rPAD).eval()
                valid_paths_embed = np.sum(np.array([self.train_environment.relation2vec[path] for path in valid_paths]),axis=1)
                # print valid_paths_embed.shape
                # valid_paths_embed = np.sum(valid_paths_embed,axis=1)
                valid_paths_embed = np.expand_dims(valid_paths_embed, axis=1)
                # print 'valid_paths_embed:', valid_paths_embed.shape

                # get noise representation
                ##########
                random_inputs = np.random.random(valid_paths_embed.shape)

                # get embedding of demo paths
                ##########
                # print 'start entity:', episode.start_entities[0]
                valid_actions = self.train_environment.get_valid_actions(episode.start_entities[0])
                valid_path_idx = [idx for idx, action_id in enumerate(paths_first_id) if action_id in valid_actions]
                # print 'valid_path_idx:',valid_path_idx
                valid_path_num = len(valid_path_idx)

                if valid_nums > self.disc_size:
                    valid_nums = self.disc_size

                if valid_path_num == 0:
                    real_inputs = real_inputs_all[:valid_nums]
                    print 'real_inputs 1:', real_inputs.shape
                elif valid_path_num >= valid_nums:
                    real_inputs_idx = valid_path_idx[:valid_nums]
                    # print 'real_inputs_idx:',real_inputs_idx
                    real_inputs = real_inputs_all[real_inputs_idx]
                    # print real_inputs
                    print 'real_inputs 2:', real_inputs.shape
                else:
                    diff = valid_nums - valid_path_num
                    print diff
                    paths_context_idx = [idx for idx, action_id in enumerate(paths_first_id) if idx not in valid_path_idx]
                    print 'paths_context_idx:', paths_context_idx
                    print 'valid_path_idx:', valid_path_idx
                    context_paths = real_inputs_all[paths_context_idx]
                    query_paths = real_inputs_all[valid_path_idx]
                    print context_paths.shape
                    print query_paths.shape
                    semantic_sim = np.sum(np.dot(np.reshape(context_paths, (-1, self.disc_embedding_size)),
                                                 np.reshape(query_paths, (-1, self.disc_embedding_size)).T), axis=1)
                    # print semantic_sim
                    padding_paths_idx = np.array(paths_context_idx)[np.argsort(-semantic_sim)[:diff]]
                    print 'padding_paths_idx:', padding_paths_idx
                    real_inputs_idx = valid_path_idx + padding_paths_idx.tolist()
                    print 'real_inputs_idx:', real_inputs_idx
                    real_inputs = real_inputs_all[real_inputs_idx]
                    print 'real_inputs 3:',real_inputs.shape

                real_inputs_padding = valid_paths_embed[valid_nums:]
                # print real_inputs_padding.shape
                real_inputs = np.concatenate((real_inputs, real_inputs_padding),axis=0)
                # print real_inputs
                # print 'real_inputs:',real_inputs.shape

                self.discriminator.update(real_inputs, random_inputs)
                self.discriminator.update(real_inputs, valid_paths_embed)
                _, gen_reward = self.discriminator.predict(real_inputs, valid_paths_embed)
                _, gen_reward_random = self.discriminator.predict(real_inputs, random_inputs)
                d_reward = gen_reward-gen_reward_random
                print 'gen_reward:',gen_reward
                print 'gen_reward_random:', gen_reward_random
                print 'D-reward:',d_reward
                if d_reward == 0:
                    raise ArithmeticError("Error in computing D-reward")
                gen_loss = max(d_reward,0)
                reshaped_reward = self.reshape_reward(gen_loss*rewards)
                train_loss = gen_loss
                # print 'reshaped_reward1:', reshaped_reward.shape
            else:  # len(valid_paths_idx) > 0:
                reshaped_reward = self.reshape_reward(rewards)
                train_loss = 0.0
                # print 'reshaped_reward2:',reshaped_reward.shape
            # print rewards
            # print reshaped_reward
            # print self.gamma


            # print candidate_paths.T[rewards]

            # print valid_paths

            # computed cumulative discounted reward
            # [batch_size, path length]
            # cum_discounted_reward = self.calc_cum_discounted_reward(rewards)  # [B, T]
            # print cum_discounted_reward
            # raw_input('----------')
            # print cum_discounted_reward
            # print cum_discounted_reward.shape
            # raw_input('----------')

            # back prop
            batch_total_loss, _ = sess.partial_run(h, [self.loss_op, self.dummy], feed_dict={self.final_reward: reshaped_reward})
            # raw_input('----------')
            # print statistics
            # hand-craft hyper-parameters for reward function
            avg_reward = np.mean(rewards)
            # now reshape the reward to [orig_batch_size, num_rollouts], I want to calculate for how many of the
            # entity pair, atleast one of the path get to the right answer
            reward_reshape = np.reshape(rewards, (self.batch_size, self.num_rollouts))  # [orig_batch, num_rollouts]
            reward_reshape = np.sum(reward_reshape, axis=1)  # [orig_batch]
            reward_reshape = (reward_reshape > 0)
            num_ep_correct = np.sum(reward_reshape)
            if np.isnan(train_loss):
                raise ArithmeticError("Error in computing loss")

            logger.info("batch_counter: {0:4d}, num_hits: {1:7.4f}, avg. reward per batch {2:7.4f}, "
                        "num_ep_correct {3:4d}, avg_ep_correct {4:7.4f}, train loss {5:7.4f}".
                        format(self.batch_counter, np.sum(rewards), avg_reward, num_ep_correct,
                               (num_ep_correct / self.batch_size),
                               train_loss))

            if self.batch_counter%self.eval_every == 0:
                with open(self.output_dir + '/scores.txt', 'a') as score_file:
                    score_file.write("Score for iteration " + str(self.batch_counter) + "\n")
                os.mkdir(self.path_logger_file + "/" + str(self.batch_counter))
                self.path_logger_file_ = self.path_logger_file + "/" + str(self.batch_counter) + "/paths"

                self.test(sess, beam=True, print_paths=False)

            logger.info('Memory usage: %s (kb)' % resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)

            gc.collect()
            if self.batch_counter >= self.total_iterations:
                break

    def test(self, sess, beam=False, print_paths=False, save_model = True, auc = False):
        batch_counter = 0
        paths = defaultdict(list)
        answers = []
        feed_dict = {}
        all_final_reward_1 = 0
        all_final_reward_3 = 0
        all_final_reward_5 = 0
        all_final_reward_10 = 0
        all_final_reward_20 = 0
        auc = 0

        total_examples = self.test_environment.total_no_examples
        for episode in tqdm(self.test_environment.get_episodes()):
            batch_counter += 1

            temp_batch_size = episode.no_examples

            self.qr = episode.get_query_relation()
            feed_dict[self.query_relation] = self.qr
            # set initial beam probs
            beam_probs = np.zeros((temp_batch_size * self.test_rollouts, 1))
            # get initial state
            state = episode.get_state()
            mem = self.agent.get_mem_shape()
            agent_mem = np.zeros((mem[0], mem[1], temp_batch_size*self.test_rollouts, mem[3]) ).astype('float32')
            previous_relation = np.ones((temp_batch_size * self.test_rollouts, ), dtype='int64') * self.relation_vocab[
                'DUMMY_START_RELATION']
            feed_dict[self.range_arr] = np.arange(temp_batch_size * self.test_rollouts)
            feed_dict[self.input_path[0]] = np.zeros(temp_batch_size * self.test_rollouts)

            ####logger code####
            if print_paths:
                self.entity_trajectory = []
                self.relation_trajectory = []
            ####################

            self.log_probs = np.zeros((temp_batch_size*self.test_rollouts,)) * 1.0

            # for each time step
            for i in range(self.path_length):
                if i == 0:
                    feed_dict[self.first_state_of_test] = True
                feed_dict[self.next_relations] = state['next_relations']
                feed_dict[self.next_entities] = state['next_entities']
                feed_dict[self.current_entities] = state['current_entities']
                feed_dict[self.prev_state] = agent_mem
                feed_dict[self.prev_relation] = previous_relation

                loss, agent_mem, test_scores, test_action_idx, chosen_relation = sess.run(
                    [self.test_loss, self.test_state, self.test_logits, self.test_action_idx, self.chosen_relation],
                    feed_dict=feed_dict)

                if beam:
                    k = self.test_rollouts
                    new_scores = test_scores + beam_probs
                    if i == 0:
                        idx = np.argsort(new_scores)
                        idx = idx[:, -k:]
                        ranged_idx = np.tile([b for b in range(k)], temp_batch_size)
                        idx = idx[np.arange(k*temp_batch_size), ranged_idx]
                    else:
                        idx = self.top_k(new_scores, k)

                    y = idx//self.max_num_actions
                    x = idx%self.max_num_actions

                    y += np.repeat([b*k for b in range(temp_batch_size)], k)
                    state['current_entities'] = state['current_entities'][y]
                    state['next_relations'] = state['next_relations'][y,:]
                    state['next_entities'] = state['next_entities'][y, :]
                    agent_mem = agent_mem[:, :, y, :]
                    test_action_idx = x
                    chosen_relation = state['next_relations'][np.arange(temp_batch_size*k), x]
                    beam_probs = new_scores[y, x]
                    beam_probs = beam_probs.reshape((-1, 1))
                    if print_paths:
                        for j in range(i):
                            self.entity_trajectory[j] = self.entity_trajectory[j][y]
                            self.relation_trajectory[j] = self.relation_trajectory[j][y]
                previous_relation = chosen_relation

                ####logger code####
                if print_paths:
                    self.entity_trajectory.append(state['current_entities'])
                    self.relation_trajectory.append(chosen_relation)
                ####################
                state = episode(test_action_idx)
                self.log_probs += test_scores[np.arange(self.log_probs.shape[0]), test_action_idx]
            if beam:
                self.log_probs = beam_probs

            ####Logger code####

            if print_paths:
                self.entity_trajectory.append(state['current_entities'])

            # ask environment for final reward
            rewards = episode.get_reward()  # [B*test_rollouts]
            reward_reshape = np.reshape(rewards, (temp_batch_size, self.test_rollouts))  # [orig_batch, test_rollouts]
            self.log_probs = np.reshape(self.log_probs, (temp_batch_size, self.test_rollouts))
            sorted_indx = np.argsort(-self.log_probs)
            final_reward_1 = 0
            final_reward_3 = 0
            final_reward_5 = 0
            final_reward_10 = 0
            final_reward_20 = 0
            AP = 0
            ce = episode.state['current_entities'].reshape((temp_batch_size, self.test_rollouts))
            se = episode.start_entities.reshape((temp_batch_size, self.test_rollouts))
            for b in range(temp_batch_size):
                answer_pos = None
                seen = set()
                pos=0
                if self.pool == 'max':
                    for r in sorted_indx[b]:
                        if reward_reshape[b,r] == self.positive_reward:
                            answer_pos = pos
                            break
                        if ce[b, r] not in seen:
                            seen.add(ce[b, r])
                            pos += 1
                if self.pool == 'sum':
                    scores = defaultdict(list)
                    answer = ''
                    for r in sorted_indx[b]:
                        scores[ce[b,r]].append(self.log_probs[b,r])
                        if reward_reshape[b,r] == self.positive_reward:
                            answer = ce[b,r]
                    final_scores = defaultdict(float)
                    for e in scores:
                        final_scores[e] = lse(scores[e])
                    sorted_answers = sorted(final_scores, key=final_scores.get, reverse=True)
                    if answer in  sorted_answers:
                        answer_pos = sorted_answers.index(answer)
                    else:
                        answer_pos = None

                if answer_pos is not None:
                    if answer_pos < 20:
                        final_reward_20 += 1
                        if answer_pos < 10:
                            final_reward_10 += 1
                            if answer_pos < 5:
                                final_reward_5 += 1
                                if answer_pos < 3:
                                    final_reward_3 += 1
                                    if answer_pos < 1:
                                        final_reward_1 += 1
                if answer_pos is None:
                    AP += 0
                else:
                    if AP == 0:
                        AP = 1.0/((answer_pos+1))
                if print_paths:
                    qr = self.train_environment.grapher.rev_relation_vocab[self.qr[b * self.test_rollouts]]
                    start_e = self.rev_entity_vocab[episode.start_entities[b * self.test_rollouts]]
                    end_e = self.rev_entity_vocab[episode.end_entities[b * self.test_rollouts]]
                    paths[str(qr)].append(str(start_e) + "\t" + str(end_e) + "\n")
                    paths[str(qr)].append("Reward:" + str(1 if answer_pos != None and answer_pos < 10 else 0) + "\n")
                    for r in sorted_indx[b]:
                        indx = b * self.test_rollouts + r
                        if rewards[indx] == self.positive_reward:
                            rev = 1
                        else:
                            rev = -1
                        answers.append(self.rev_entity_vocab[se[b,r]]+'\t'+ self.rev_entity_vocab[ce[b,r]]+'\t'+ str(self.log_probs[b,r])+'\n')
                        paths[str(qr)].append(
                            '\t'.join([str(self.rev_entity_vocab[e[indx]]) for e in
                                       self.entity_trajectory]) + '\n' + '\t'.join(
                                [str(self.rev_relation_vocab[re[indx]]) for re in self.relation_trajectory]) + '\n' + str(
                                rev) + '\n' + str(
                                self.log_probs[b, r]) + '\n___' + '\n')
                    paths[str(qr)].append("#####################\n")

            all_final_reward_1 += final_reward_1
            all_final_reward_3 += final_reward_3
            all_final_reward_5 += final_reward_5
            all_final_reward_10 += final_reward_10
            all_final_reward_20 += final_reward_20
            auc += AP

        all_final_reward_1 /= total_examples
        all_final_reward_3 /= total_examples
        all_final_reward_5 /= total_examples
        all_final_reward_10 /= total_examples
        all_final_reward_20 /= total_examples
        auc /= total_examples
        if save_model:
            if all_final_reward_10 >= self.max_hits_at_10:
                self.max_hits_at_10 = all_final_reward_10
                self.save_path = self.model_saver.save(sess, self.model_dir + "model" + '.ckpt')

        if print_paths:
            logger.info("[ printing paths at {} ]".format(self.output_dir+'/test_beam/'))
            for q in paths:
                j = q.replace('/', '-')
                with codecs.open(self.path_logger_file_ + '_' + j, 'a', 'utf-8') as pos_file:
                    for p in paths[q]:
                        pos_file.write(p)
            with open(self.path_logger_file_ + 'answers', 'w') as answer_file:
                for a in answers:
                    answer_file.write(a)

        with open(self.output_dir + '/scores.txt', 'a') as score_file:
            score_file.write("Hits@1: {0:7.4f}".format(all_final_reward_1))
            score_file.write("\n")
            score_file.write("Hits@3: {0:7.4f}".format(all_final_reward_3))
            score_file.write("\n")
            score_file.write("Hits@5: {0:7.4f}".format(all_final_reward_5))
            score_file.write("\n")
            score_file.write("Hits@10: {0:7.4f}".format(all_final_reward_10))
            score_file.write("\n")
            score_file.write("Hits@20: {0:7.4f}".format(all_final_reward_20))
            score_file.write("\n")
            score_file.write("MRR: {0:7.4f}".format(auc))
            score_file.write("\n")
            score_file.write("\n")

        logger.info("Hits@1: {0:7.4f}".format(all_final_reward_1))
        logger.info("Hits@3: {0:7.4f}".format(all_final_reward_3))
        logger.info("Hits@5: {0:7.4f}".format(all_final_reward_5))
        logger.info("Hits@10: {0:7.4f}".format(all_final_reward_10))
        logger.info("Hits@20: {0:7.4f}".format(all_final_reward_20))
        logger.info("MRR: {0:7.4f}".format(auc))

    def top_k(self, scores, k):
        scores = scores.reshape(-1, k * self.max_num_actions)  # [B, (k*max_num_actions)]
        idx = np.argsort(scores, axis=1)
        idx = idx[:, -k:]  # take the last k highest indices # [B , k]
        return idx.reshape((-1))

if __name__ == '__main__':

    # read command line options by options.py
    options = read_options()
    # Set logging
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter('%(asctime)s: [ %(message)s ]', '%m/%d/%Y %I:%M:%S %p')
    console = logging.StreamHandler()
    console.setFormatter(fmt)
    logger.addHandler(console)
    logfile = logging.FileHandler(options['log_file_name'], 'w')
    logfile.setFormatter(fmt)
    logger.addHandler(logfile)
    # read the vocab files, it will be used by many classes hence global scope
    logger.info('reading vocab files...')
    options['relation_vocab'] = json.load(open(options['vocab_dir'] + '/relation_vocab.json'))
    options['entity_vocab'] = json.load(open(options['vocab_dir'] + '/entity_vocab.json'))
    logger.info('Reading mid to name map')
    logger.info('Done..')
    logger.info('Total number of entities {}'.format(len(options['entity_vocab'])))
    logger.info('Total number of relations {}'.format(len(options['relation_vocab'])))
    save_path = ''
    os.environ["CUDA_VISIBLE_DEVICES"] = '3'  # use GPU with ID
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.8  # maximun alloc gpu80% of MEM  
    config.gpu_options.allow_growth = False
    config.log_device_placement = False

    # Training
    if not options['load_model']:
        trainer = Trainer(options)
        # need to add Discriminator here
        # print trainer.agent.embedding_size
        # print trainer.agent.hidden_size
        # raw_input('----------')
        # print options['embedding_size']
        # print options['data_input_dir']
        # raw_input('---------')

        # disc_embedding_size = 2 * options['embedding_size']
        # relationPath = options['data_input_dir']



        with tf.Session(config=config) as sess:
            sess.run(trainer.initialize())

            # print trainer.action_idx[0].eval()
            # raw_input('----------')
            # saver = tf.train.Saver()
            # saver.restore(sess, relationPath + 'models/pre_train_model_')
            # we need to add the pre-trained embeddings by ourselves
            # trainer.initialize_pretrained_embeddings(sess=sess)

            suc = trainer.pre_train(sess=sess)
            retry_times = 0
            while (suc is False) and (retry_times <= 5):
                # print('suc:',suc)
                retry_times += 1
                print('retry times:', retry_times)
                sess.run(trainer.initialize())
                suc = trainer.pre_train(sess=sess)

            trainer.train(sess)
            save_path = trainer.save_path
            path_logger_file = trainer.path_logger_file
            output_dir = trainer.output_dir

        tf.reset_default_graph()
    else:
        logger.info("Skipping training")
        logger.info("Loading model from {}".format(options["model_load_dir"]))

    # Test
    trainer = Trainer(options)
    if options['load_model']:
        save_path = options['model_load_dir']
        path_logger_file = trainer.path_logger_file
        output_dir = trainer.output_dir
    with tf.Session(config=config) as sess:
        trainer.initialize(restore=save_path, sess=sess)
        trainer.test_rollouts = 100
        os.mkdir(path_logger_file + "/" + "test_beam")
        trainer.path_logger_file_ = path_logger_file + "/" + "test_beam" + "/paths"

        # save score.txt for Hits@K and accuracy
        with open(output_dir + '/scores.txt', 'a') as score_file:
            score_file.write("Test (beam) scores with best model from " + save_path + "\n")

        trainer.test_environment = trainer.test_test_environment
        trainer.test_environment.test_rollouts = 100
        trainer.test(sess, beam=True, print_paths=True, save_model=False)

        # print options['nell_evaluation']
        if options['nell_evaluation'] == 1:
            current_relation = options['data_input_dir'].split('/')[-2]
            print 'current_relation:', current_relation
            mean_ap, mean_hit_1, mean_hit_3, mean_hit_10, mean_mrr = nell_eval(
                path_logger_file + "/" + "test_beam/" + "pathsanswers", trainer.data_input_dir + '/sort_test.pairs')
            with open(trainer.output_dir + '/' + current_relation + '_nell_eval.txt', 'a') as nell_eval_file:
                # nell_eval_file.write('MINERVA MAP: {}'.format(mean_ap))
                nell_eval_file.write(
                    'RL MAP: ' + str(mean_ap) + '\n' + 'HITS@1: ' + str(mean_hit_1) + '\n' + 'HITS@3: ' + str(
                        mean_hit_3) + '\n' + 'HITS@10: ' + str(mean_hit_10) + '\n' + 'MRR: ' + str(mean_mrr) + '\n')
