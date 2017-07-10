import ipdb
import math
import random
import numpy as np
import tensorflow as tf


def e_greedy_sampler(sess, best_action_picker, s1_tf, training, epsilon=0.05):
    def __sampler__(actions, state):
        if random.random() < epsilon:
            return actions.sample()
        else:
            return np.argmax(sess.run(best_action_picker, feed_dict={
                s1_tf: state,
                training: False})[0])
    return __sampler__


def greedy_sampler(sess, best_action_picker, s1_tf, training):
    def __sampler__(actions, state):
        return np.argmax(sess.run(best_action_picker, feed_dict={
            s1_tf: state,
            training: False})[0])
    return __sampler__


