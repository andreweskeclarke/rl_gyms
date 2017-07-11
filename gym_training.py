import os
import gym
import ipdb
import math
import time
import random
import pickle
import argparse
import collections
import numpy as np
import tensorflow as tf

from cartpole_runner import *
from action_samplers import *
from models import N_DIM_ACTIONS, N_DIM_STATE, ddqn_mlp, ladder_mlp
from file_utils import *

config = tf.ConfigProto()
config.gpu_options.allow_growth=True

print('#####################################################################################')
print('# Training')
print('#####################################################################################')
parser = argparse.ArgumentParser(description='Train an RL agent to learn Cart Pole')
parser.add_argument('--lr', help='learning rates to try', metavar='N', type=float, nargs='+')
parser.add_argument('--n_hidden', help='n hidden nodes', type=int)
parser.add_argument('--dir', help='output directory')
parser.add_argument('--use_ladder', dest='use_ladder', action='store_true')
parser.add_argument('--experiments', help='To help batching experiments, comma separated list ' +\
        'of experiment indexes', type=str)
args = parser.parse_args()

# Training - Setup
DISCOUNT = 0.99
N_EPISODES = 1000
BATCH_SIZE = 16
N_STEPS_BETWEEN_TARGET_UPDATES = 1
N_EPISODES_PER_LOG = 20
EXPERIENCE_REPLAY_BUFFER_SIZE = 5000
learning_rates = [1e-3] if args.lr is None else args.lr
n_hidden = 100 if args.n_hidden is None else args.n_hidden
progress_dir = 'scratch' if args.dir is None else args.dir
use_ladder = args.use_ladder
progress_dir = os.path.join('models', progress_dir)
env = gym.make('CartPole-v000')
start_time = time.time()
encoded_actions = np.array([[1, 0], [0, 1]])
experiment_indexes = [x for x in range(0, 25)]
if args.experiments is not None:
    experiment_indexes = [int(x) for x in args.experiments.split(',')]

print('Using learning rates: %s' % str(learning_rates))
print('Running experiments %s' % str(experiment_indexes))
print('Using network with %d hidden layers' % n_hidden)
print('Using ladder network: %s' % use_ladder)
print('#####################################################################################')

# Repeat the experiment for different learning rates
for learning_rate in learning_rates:
    print('Learning rate %f' % learning_rate)
    # Run the experiment 100 times to get a good training graph
    for curr_experiment in experiment_indexes:
        print('Running experiment %d... (%f s)' % (curr_experiment + 1, time.time() - start_time))
        test_lengths = np.zeros([int(N_EPISODES/N_EPISODES_PER_LOG)])
        lengths = np.zeros([int(N_EPISODES)])
        losses = np.zeros([int(N_EPISODES)])
        experiences = collections.deque(maxlen=EXPERIENCE_REPLAY_BUFFER_SIZE)

        with tf.Session(config=config) as sess:
            s1_tf = tf.placeholder("float", [None, N_DIM_STATE], name='s1_inputs')
            a1_tf = tf.placeholder("float", [None, N_DIM_ACTIONS], name='a1_inputs')
            r1_tf = tf.placeholder("float", [None, 1], name='r1_inputs')
            s2_tf = tf.placeholder("float", [None, N_DIM_STATE], name='s2_inputs')
            layer_sizes = [N_DIM_STATE, n_hidden, n_hidden, N_DIM_ACTIONS]  # layer sizes
            if use_ladder:
                loss, train_op, best_action_picker, updater, training, tf_debug = \
                    ladder_mlp(s1_tf, a1_tf, r1_tf, s2_tf, DISCOUNT, learning_rate, layers)
            else:
                loss, train_op, best_action_picker, updater, training, tf_debug = \
                    ddqn_mlp(s1_tf, a1_tf, r1_tf, s2_tf, DISCOUNT, learning_rate, layers)
            sess.run(tf.global_variables_initializer())
            action_sampler = e_greedy_sampler(sess, best_action_picker, s1_tf, training)
            saver = tf.train.Saver()

            # Train online Q-learning algorithm over N_EPISODES_PER_LOG00 episodes
            best_length = -float('inf')
            n_updates_since_updating_target_net = 0
            for curr_episode in range(0, N_EPISODES):
                # Log current status
                if (curr_episode+1) % N_EPISODES_PER_LOG == 0:
                    print('Exp %d - ep. %d: test length (%f) train length (%f) loss (%f) %f s' %
                          (curr_experiment + 1,
                           curr_episode + 1,
                           test_lengths[int(curr_episode / N_EPISODES_PER_LOG)],
                           np.mean(lengths[curr_episode - N_EPISODES_PER_LOG:curr_episode]),
                           losses[curr_episode - 1],
                           time.time() - start_time))

                # Run one episode
                state = env.reset()
                done = False
                curr_episode_length = 0
                while not done:
                    action = action_sampler(env.action_space, np.array(state, ndmin=2))
                    next_state, _, done, _ = env.step(action)
                    reward = -1 if done else 0
                    experiences.append((state, encoded_actions[action, :], reward, next_state))
                    curr_episode_length += 1

                    # Train network from experiences saved in the replay buffer
                    exps = [experiences[random.randint(0, len(experiences)-1)] for i in range(0, BATCH_SIZE)]
                    sess.run(train_op, feed_dict={
                        s1_tf: np.array([s1 for s1, a1, r1, s2 in exps]),
                        a1_tf: np.array([a1 for s1, a1, r1, s2 in exps]),
                        r1_tf: np.array([r1 for s1, a1, r1, s2 in exps], ndmin=2).T,
                        s2_tf: np.array([s2 for s1, a1, r1, s2 in exps]),
                        training: True
                    })
                    state = next_state

                    # DDQN - Update the target network every N updates
                    n_updates_since_updating_target_net += 1
                    if n_updates_since_updating_target_net >= N_STEPS_BETWEEN_TARGET_UPDATES:
                        n_updates_since_updating_target_net = 0
                        updater(sess)

                lengths[curr_episode] = curr_episode_length
                losses[curr_episode] = sess.run(loss, feed_dict={
                    s1_tf: np.array([s1 for s1, a1, r1, s2 in experiences]),
                    a1_tf: np.array([a1 for s1, a1, r1, s2 in experiences]),
                    r1_tf: np.array([r1 for s1, a1, r1, s2 in experiences], ndmin=2).T,
                    s2_tf: np.array([s2 for s1, a1, r1, s2 in experiences]),
                    training: True
                })
                if curr_episode % N_EPISODES_PER_LOG == 0:
                    if tf_debug is not None:
                        print(sess.run(tf_debug, feed_dict={
                            s1_tf: np.array([s1 for s1, a1, r1, s2 in experiences]),
                            a1_tf: np.array([a1 for s1, a1, r1, s2 in experiences]),
                            r1_tf: np.array([r1 for s1, a1, r1, s2 in experiences], ndmin=2).T,
                            s2_tf: np.array([s2 for s1, a1, r1, s2 in experiences]),
                            training: True
                        }))
                    _, episode_lengths, episode_rewards = \
                        run_n_cartpole_simulations(
                            100,
                            greedy_sampler(sess, best_action_picker, s1_tf, training),
                            env)
                    test_lengths[int(curr_episode / N_EPISODES_PER_LOG)] = np.mean(np.array(episode_lengths))
                    saved_progress_path = lr_filename(progress_dir, 'test_lengths_%d' % curr_experiment, learning_rate, 'npy')
                    if not os.path.exists(os.path.dirname(saved_progress_path)):
                        os.makedirs(os.path.dirname(saved_progress_path))
                    np.save(saved_progress_path, test_lengths)
                    saved_model_path = os.path.join(progress_dir, 'best_model_%d' % np.mean(np.array(episode_lengths)))
                    if not os.path.exists(os.path.dirname(saved_model_path)):
                        os.makedirs(os.path.dirname(saved_model_path))
                    saver.save(sess, saved_model_path)

        np.save(lr_filename(progress_dir, 'lengths_%d' % curr_experiment, learning_rate, 'npy'), lengths)
        np.save(lr_filename(progress_dir, 'losses_%d' % curr_experiment, learning_rate, 'npy'), losses)
        print('Saved! (%f)' % (time.time() - start_time))


