# -*- coding: utf-8 -*-
from __future__ import division
import tensorflow as tf
import numpy as np
import sys

from networks import policy_nn
from utils import *

relation = sys.argv[1]


class Generator(object):
    """docstring for Generator"""

    def __init__(self, learning_rate=0.001):
        self.initializer = tf.contrib.layers.xavier_initializer()
        # hyper-parameters are settled in utils.py
        # default state_dim = 200, action_space = 400(total 400 relations)
        with tf.variable_scope('generator'):
            self.state = tf.placeholder(tf.float32, [None, state_dim], name='state')
            self.action = tf.placeholder(tf.int32, [None], name='action')
            self.target = tf.placeholder(tf.float32, name='target')
            # action probability (the output of softmax layer)
            self.action_prob = policy_nn(self.state, state_dim, action_space, self.initializer)

            # change dtype from tf.one_hot to tf.bool
            # tf.one_hot action*action_space
            action_mask = tf.cast(tf.one_hot(self.action, depth=action_space), tf.bool)
            # tensor returned corresponding to True values in mask
            self.picked_action_prob = tf.boolean_mask(self.action_prob, action_mask)
            # policy grdient
            self.loss = tf.reduce_sum(-tf.log(self.picked_action_prob) * self.target) + sum(
                tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES, scope='generator'))
            self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)  # AdamOptimizer
            self.train_op = self.optimizer.minimize(self.loss)

    def predict(self, state, sess=None):
        sess = sess or tf.get_default_session()
        return sess.run(self.action_prob, {self.state: state})

    def update(self, state, target, action, sess=None):
        sess = sess or tf.get_default_session()
        feed_dict = {self.state: state, self.target: target, self.action: action}
        _, loss = sess.run([self.train_op, self.loss], feed_dict)
        return loss
