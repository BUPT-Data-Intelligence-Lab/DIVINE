# -*- coding: utf-8 -*-
from __future__ import division

import collections
import numpy as np
import sys
import tensorflow as tf
import time
import math
from BFS.BFS import BFS
from BFS.KB import KB
from itertools import count
from networks import policy_nn, discriminator_nn
from utils import *

relation = sys.argv[1]
# episodes = int(sys.argv[2])
# dataPath = '../NELL-995/' was settled in utils.py
graphpath = dataPath + 'tasks/' + relation + '/' + 'graph.txt'
relationPath = dataPath + 'tasks/' + relation + '/' + 'train_pos'
demoPath = dataPath + 'tasks/' + relation + '/' + 'demo_path.txt'
negPath = dataPath + 'tasks/' + relation + '/' + 'neg_path.txt'

LAMBDA = 5  # Gradient penalty lambda hyper-parameter.


class Discriminator(object):
    """docstring for Discriminator"""

    def __init__(self, batch_size,learning_rate=0.001):
        self.initializer = tf.contrib.layers.xavier_initializer()
        # hyper-parameters are settled in utils.py
        # default state_dim = 200, action_space = 400(total 400 relations)
        with tf.variable_scope('discriminator') as scope:
            # inputs are embeddings not paths
            self.task = tf.placeholder(tf.string)
            self.real_inputs = tf.placeholder(tf.float32, [batch_size, 1, embedding_dim], name='real_inputs')

            self.fake_inputs = tf.placeholder(tf.float32, [batch_size, 1, embedding_dim], name='fake_inputs')
            # normalize_real = tf.nn.l2_normalize(real_inputs.reshape((bach_size*embedding_dim)),0)
            # normalize_fake = tf.nn.l2_normalize(fake_inputs.reshape((bach_size*embedding_dim)),0)
            # self.gen_reward = tf.reduce_sum(tf.multiply(normalize_real,normalize_fake))

            disc_real = discriminator_nn(self.real_inputs, embedding_dim, self.task, self.initializer)
            scope.reuse_variables()
            disc_fake = discriminator_nn(self.fake_inputs, embedding_dim, self.task, self.initializer)
            # original critic loss
            original_disc_cost = tf.reduce_mean(disc_fake) - tf.reduce_mean(disc_real)

            # WGAN lipschitz-penalty
            alpha = tf.random_uniform(shape=[batch_size, 1, 1], minval=0., maxval=1.)
            differences = self.fake_inputs - self.real_inputs
            interpolate = self.real_inputs + (alpha * differences)

            # tf.gradients(ys,xs)
            grad = tf.gradients(discriminator_nn(interpolate, embedding_dim, self.task, self.initializer)[0], [interpolate])[0]
            slopes = tf.sqrt(tf.reduce_sum(tf.square(grad), reduction_indices=[1, 2]))
            gradient_penalty = tf.reduce_mean((slopes - 1.) ** 2)

            # total loss for D and G
            self.disc_cost = (original_disc_cost + LAMBDA * gradient_penalty)
            self.gen_reward = tf.reduce_mean(disc_fake)

            self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
            self.train_op = self.optimizer.minimize(self.disc_cost)

    def predict(self, real, fake, sess=None):
        sess = sess or tf.get_default_session()
        return sess.run([self.disc_cost, self.gen_reward],
                        {self.task: 'test', self.real_inputs: real, self.fake_inputs: fake})

    def update(self, real, fake, sess=None):
        sess = sess or tf.get_default_session()
        _, loss = sess.run([self.train_op, self.disc_cost],
                           {self.task: 'train', self.real_inputs: real, self.fake_inputs: fake})
        return loss


def test(epochs=10000):  # iterations to train for in WGAN-GP is 200000
    tf.reset_default_graph()

    env = Env(dataPath)

    f = open(demoPath)
    real_inputs = np.array([env.path_embedding(item.strip().rsplit(' -> ')) for item in f.readlines()])
    real_inputs = real_inputs.astype('float32')  # (paths_number,1,embedding_dim)
    f.close()

    f = open(negPath)
    fake_inputs = np.array([env.path_embedding(item.strip().rsplit(' -> ')) for item in f.readlines()])
    fake_inputs = fake_inputs.astype('float32')
    f.close()

    # BATCH_SIZE = len(real_inputs)
    BATCH_SIZE = 5

    real_inputs = real_inputs[:BATCH_SIZE]
    fake_inputs = fake_inputs[:BATCH_SIZE]

    print real_inputs.shape
    # print fake_inputs.shape

    random_inputs = np.random.random(real_inputs.shape)
    print random_inputs.shape

    discriminator = Discriminator(BATCH_SIZE)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        disc_cost, gen_reward = discriminator.predict(real_inputs, random_inputs)
        print('----------')
        print('disc_cost:', disc_cost)
        print('gen_reward:', gen_reward)
        # print w1
        # print w2
        for ind in range(epochs):
            random_inputs = np.random.random(real_inputs.shape)
            discriminator.update(real_inputs, random_inputs)
            if ind % (epochs / 10) == 0:
                disc_cost, gen_reward= discriminator.predict(real_inputs, random_inputs)
                if math.isnan(disc_cost):
                    print('----- retry -----')
                    return False

                print('----------')
                print('disc_cost:', disc_cost)
                print('gen_reward:', gen_reward)

        print('----------real_inputs----------')
        disc_cost, gen_reward = discriminator.predict(real_inputs, real_inputs)
        print('disc_cost:', disc_cost)
        print('gen_reward:', gen_reward)

        print('----------random_inputs----------')
        disc_cost, gen_reward= discriminator.predict(real_inputs, random_inputs)
        print('disc_cost:', disc_cost)
        print('gen_reward:', gen_reward)

        print('----------fake_inputs----------')
        disc_cost, gen_reward= discriminator.predict(real_inputs, fake_inputs)
        print('disc_cost:', disc_cost)
        print('gen_reward:', gen_reward)

        return gen_reward


if __name__ == '__main__':
    suc = test()
    retry_times = 0
    while (suc is False) and (retry_times <=5):
        # print('suc:',suc)
        retry_times += 1
        print('retry times:',retry_times)
        suc = test()
