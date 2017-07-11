import ipdb
import math
import numpy as np
import tensorflow as tf

# N_DIM_STATE = 4
# N_DIM_ACTIONS = 2
N_DIM_STATE = 210*160
N_DIM_ACTIONS = 9

def batch_norm_init(inits, size, name):
    return tf.Variable(inits * tf.ones([size]), name=name)

def weight_init(shape, name):
    return tf.Variable(tf.random_normal(shape, stddev=math.sqrt(shape[0])), name=name)

def batch_normalization(batch, mean=None, var=None):
    if mean is None or var is None:
        mean, var = tf.nn.moments(batch, axes=[0])
    return (batch - mean) / tf.sqrt(var + tf.constant(1e-9))

def update_batch_normalization(batch, l, bn_assigns, running_mean, running_var, ewma):
    mean, var = tf.nn.moments(batch, axes=[0])
    assign_mean = running_mean[l - 1].assign(mean)
    assign_var = running_var[l - 1].assign(var)
    bn_assigns.append(ewma.apply([running_mean[l - 1], running_var[l - 1]]))
    with tf.control_dependencies([assign_mean, assign_var]):
        return (batch - mean) / tf.sqrt(var + 1e-10)

def ddqn(s1, a1, r1, s2, discount, learning_rate, layers, q_values_fun_builder):
    training = tf.placeholder(tf.bool)
    n_data = tf.shape(s1)[0]
    # DDQN - Find best value using the up to date Q function, but estimate it's value from our target Q function.
    targets, _, bn_assigns, target_weights, _ = q_values_fun_builder(s2, training)
    best_action = tf.argmax(targets, axis=1)
    # Cases when the second action is picked
    second_action_is_best = tf.cast(best_action, dtype=bool)
    # DDQN Pick action with Q_1, score with Q_target
    ddqn_target_scores, _, _, ddqn_target_weights, _ = q_values_fun_builder(s2, training)
    target_scores = tf.where(
        second_action_is_best,
        discount*ddqn_target_scores[:, 1],
        discount*ddqn_target_scores[:, 0])
    # Remove future score prediction if end of episode
    future_score = tf.where(
        tf.equal(r1, -1*tf.ones(tf.shape(r1))),
        tf.zeros(tf.shape(r1)),
        tf.reshape(target_scores, [-1, 1]))

    target_q_valuez = tf.concat([r1 + future_score for _ in range(N_DIM_ACTIONS)], 1)
    all_ones = tf.concat([tf.ones([n_data, 1]) for _ in range(N_DIM_ACTIONS)], 1)
    predicted_q_values, _, _, online_weights, _ = q_values_fun_builder(s1, training)
    target_q_values = tf.where(
        tf.equal(a1, all_ones),
        target_q_valuez,
        predicted_q_values)

    best_action_picker, u_loss, bn_assigns, _, tf_debug_var = q_values_fun_builder(s1, training, online_weights)
    u_loss = (u_loss * tf.constant(1/100))
    supervised_loss = tf.reduce_mean(tf.square(tf.stop_gradient(target_q_values) - predicted_q_values))
    loss = supervised_loss + u_loss
    training_vars = []
    for w_key, weights in online_weights.items():
        training_vars = training_vars + weights
    opt = tf.train.AdamOptimizer(learning_rate=learning_rate)
    train_op = opt.minimize(loss, var_list=training_vars)

    target_updaters = []
    for w_key, weights in online_weights.items():
        for w_index in range(len(weights)):
            target_updaters.append(
                tf.assign(target_weights[w_key][w_index],
                          online_weights[w_key][w_index]))

    updaters = []
    for w_key, weights in online_weights.items():
        for w_index in range(len(weights)):
            updaters.append(
                tf.assign(ddqn_target_weights[w_key][w_index],
                          online_weights[w_key][w_index]))

    def updater(sess):
        for u in updaters:
            sess.run(u)

    # add the updates of batch normalization statistics to train_step
    network_updates = tf.group(*(bn_assigns + target_updaters))
    with tf.control_dependencies([train_op]):
        train_op = tf.group(network_updates)

    return loss, \
           train_op, \
           best_action_picker, \
           updater, \
           training, \
           None


def ddqn_mlp(s1, a1, r1, s2, discount, learning_rate, layer_sizes):
    n_data = tf.shape(s1)[0]

    # Q-Values from a ladder network
    def q_values(state1, training, weights=None):
        L = len(layer_sizes) - 1  # number of layers
        shapes = [s for s in zip(layer_sizes[:-1], layer_sizes[1:])]  # shapes of linear layers
        if weights is None:
            weights = {
                'Encoder_w': [weight_init(s, 'Encoder_w') for s in shapes],  # Encoder weights
                'beta': [batch_norm_init(0.0, layer_sizes[l+1], 'beta') for l in range(L)],
                'gamma': [batch_norm_init(1.0, layer_sizes[l+1], 'gamma') for l in range(L)]
            }

        # Relative importance of each layer
        running_mean = [tf.Variable(tf.constant(0.0, shape=[l]), name='running_mean', trainable=False)
                        for l in layer_sizes[1:]]
        running_var = [tf.Variable(tf.constant(1.0, shape=[l]), name='running_var', trainable=False)
                       for l in layer_sizes[1:]]
        ewma = tf.train.ExponentialMovingAverage(decay=0.99)  # to calculate the moving averages of mean and variance
        bn_assigns = []  # this list stores the updates to be made to average mean and variance
        # to store the pre-activation, activation, mean and variance for each layer
        d = {'z': {}, 'm': {}, 'v': {}, 'h': {}}
        h = state1
        d['z'][0] = h
        for l in range(1, L + 1):
            print("Layer ", l, ": ", layer_sizes[l - 1], " -> ", layer_sizes[l])
            d['h'][l - 1] = h
            z_pre = tf.matmul(h, weights['Encoder_w'][l - 1])  # pre-activation
            m, v = tf.nn.moments(z_pre, axes=[0])

            # if training:
            def training_batch_norm():
                return update_batch_normalization(z_pre, l, bn_assigns, running_mean, running_var, ewma)

            # else:
            def eval_batch_norm():
                mean = ewma.average(running_mean[l - 1])
                var = ewma.average(running_var[l - 1])
                z = batch_normalization(z_pre, mean, var)
                return z

            z = tf.cond(training, training_batch_norm, eval_batch_norm)
            if l == L:
                h = tf.nn.softmax(weights['gamma'][l - 1] * (z + weights["beta"][l - 1]))
            else:
                h = tf.nn.relu(z + weights["beta"][l - 1])
            d['z'][l] = z
            d['m'][l], d['v'][l] = m, v
        d['h'][l] = h
        return h, tf.Variable(tf.constant(0.0)), bn_assigns, weights, None

    return ddqn(s1, a1, r1, s2, discount, learning_rate, layer_sizes, q_values)


# https://github.com/rinuboney/ladder/blob/master/ladder.py
def ladder_mlp(s1, a1, r1, s2, discount, learning_rate, layer_sizes):
    # Q-Values from a ladder network
    def q_values(state1, training, weights=None):
        L = len(layer_sizes) - 1  # number of layers
        shapes = [s for s in zip(layer_sizes[:-1], layer_sizes[1:])]  # shapes of linear layers
        if weights is None:
            weights = {
                'Encoder_w': [weight_init(s, 'Encoder_w') for s in shapes],  # Encoder weights
                'Decoder_w': [weight_init(s[::-1], 'Decoder_w') for s in shapes],  # Decoder weights
                'beta': [batch_norm_init(0.0, layer_sizes[l+1], 'beta') for l in range(L)],
                'gamma': [batch_norm_init(1.0, layer_sizes[l+1], 'gamma') for l in range(L)]
            }

        # Relative importance of each layer
        denoising_cost = [10.0, 2.5, 0.0, 0.0]
        running_mean = [tf.Variable(tf.constant(0.0, shape=[l]), name='running_mean', trainable=False)
                        for l in layer_sizes[1:]]
        running_var = [tf.Variable(tf.constant(1.0, shape=[l]), name='running_var', trainable=False)
                       for l in layer_sizes[1:]]

        ewma = tf.train.ExponentialMovingAverage(decay=0.99)  # to calculate the moving averages of mean and variance
        bn_assigns = []  # this list stores the updates to be made to average mean and variance

        def encoder(inputs, noise_std):
            # add noise to input
            h = inputs + tf.random_normal(tf.shape(inputs)) * noise_std
            # to store the pre-activation, activation, mean and variance for each layer
            d = {'z': {}, 'm': {}, 'v': {}, 'h': {}}
            d['z'][0] = h
            for l in range(1, L + 1):
                print("Layer ", l, ": ", layer_sizes[l - 1], " -> ", layer_sizes[l])
                d['h'][l - 1] = h
                z_pre = tf.matmul(h, weights['Encoder_w'][l - 1])  # pre-activation
                m, v = tf.nn.moments(z_pre, axes=[0])

                # if training:
                def training_batch_norm():
                    # Training batch normalization
                    # batch normalization for labeled and unlabeled examples is performed separately
                    if noise_std > 0:
                        # Corrupted encoder
                        # batch normalization + noise
                        z = batch_normalization(z_pre, m, v)
                        z += tf.random_normal(tf.shape(z_pre)) * noise_std
                    else:
                        # Clean encoder
                        # batch normalization + update the average mean and variance using batch
                        # mean and variance of labeled examples
                        z = update_batch_normalization(z_pre, l, bn_assigns, running_mean, running_var, ewma)
                    return z

                # else:
                def eval_batch_norm():
                    # Evaluation batch normalization
                    # obtain average mean and variance and use it to normalize the batch
                    mean = ewma.average(running_mean[l - 1])
                    var = ewma.average(running_var[l - 1])
                    z = batch_normalization(z_pre, mean, var)
                    return z

                # perform batch normalization according to value of boolean "training" placeholder:
                z = tf.cond(training, training_batch_norm, eval_batch_norm)

                if l == L:
                    # use softmax activation in output layer
                    h = tf.nn.softmax(weights['gamma'][l - 1] * (z + weights["beta"][l - 1]))
                else:
                    # use ReLU activation in hidden layers
                    h = tf.nn.relu(z + weights["beta"][l - 1])
                d['z'][l] = z
                d['m'][l], d['v'][l] = m, v  # save mean and variance of unlabeled examples for decoding

            d['h'][l] = h
            return h, d

        print("=== Corrupted Encoder ===")
        y_c, corr = encoder(state1, 0.1)

        print("=== Clean Encoder ===")
        y, clean = encoder(state1, 0.0)  # 0.0 -> do not add noise

        print("=== Decoder ===")

        def g_gauss(z_c, u, size):
            wi = lambda inits, name: tf.Variable(inits * tf.ones([size]), name=name)
            a1 = wi(0., 'a1')
            a2 = wi(1., 'a2')
            a3 = wi(0., 'a3')
            a4 = wi(0., 'a4')
            a5 = wi(0., 'a5')

            a6 = wi(0., 'a6')
            a7 = wi(1., 'a7')
            a8 = wi(0., 'a8')
            a9 = wi(0., 'a9')
            a10 = wi(0., 'a10')

            mu = a1 * tf.sigmoid(a2 * (u + tf.constant(1e-9)) + a3) + a4 * u + a5
            v = a6 * tf.sigmoid(a7 * (u + tf.constant(1e-9)) + a8) + a9 * u + a10

            z_est = (z_c - mu) * v + mu
            return z_est

        # Decoder
        z_est = {}
        d_cost = []  # to store the denoising cost of all layers
        for l in range(L, -1, -1):
            print("Layer ", l, ": ", layer_sizes[l+1] if l+1 < len(layer_sizes) else None,
                  " -> ", layer_sizes[l], ", denoising cost: ", denoising_cost[l])
            z, z_c = clean['z'][l], corr['z'][l]
            m = clean['m'].get(l, 0)
            v = clean['v'].get(l, 1-1e-10) + tf.constant(1e-9)
            if l == L:
                u = y_c
            else:
                u = tf.matmul(z_est[l+1], weights['Decoder_w'][l])
            u = batch_normalization(u)
            z_est[l] = g_gauss(z_c, u, layer_sizes[l])
            z_est_bn = (z_est[l] - m) / v
            # append the cost of this layer to d_cost
            d_cost.append((tf.reduce_mean(tf.reduce_sum(tf.square(z_est_bn - z), 1)) / layer_sizes[l]) * denoising_cost[l])

        # calculate total unsupervised cost by adding the denoising cost of all layers
        unsupervised_cost = tf.add_n(d_cost)

        return y, unsupervised_cost, bn_assigns, weights, None
    
    return ddqn(s1, a1, r1, s2, discount, learning_rate, layer_sizes, q_values)
