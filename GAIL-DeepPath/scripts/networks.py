import tensorflow as tf


def policy_nn(state, state_dim, action_dim, initializer):
    w1 = tf.get_variable('W1', [state_dim, 512], initializer=initializer,
                         regularizer=tf.contrib.layers.l2_regularizer(0.01))
    b1 = tf.get_variable('b1', [512], initializer=tf.constant_initializer(0.0))
    h1 = tf.nn.relu(tf.matmul(state, w1) + b1)
    w2 = tf.get_variable('w2', [512, 1024], initializer=initializer, regularizer=tf.contrib.layers.l2_regularizer(0.01))
    b2 = tf.get_variable('b2', [1024], initializer=tf.constant_initializer(0.0))
    h2 = tf.nn.relu(tf.matmul(h1, w2) + b2)
    w3 = tf.get_variable('w3', [1024, action_dim], initializer=initializer,
                         regularizer=tf.contrib.layers.l2_regularizer(0.01))
    b3 = tf.get_variable('b3', [action_dim], initializer=tf.constant_initializer(0.0))
    action_prob = tf.nn.softmax(tf.matmul(h2, w3) + b3)
    return action_prob


def discriminator_nn(inputs, embedding_dim, task, initializer):  # task may be 'train' or others
    # Input Layer
    # Reshape X to 4-D tensor: [batch_size, width, height, channels]
    height = inputs.shape[0]
    input_layer = tf.reshape(inputs, [-1, height, embedding_dim, 1])
    # print 'input_layer:', input_layer

    # convolution #1
    # [filter_height, filter_width, in_channels, out_channels]
    conv1_w = tf.get_variable('conv1_w', [3, 5, 1, 1], initializer=initializer)
    # print conv1_w
    conv1 = tf.nn.conv2d(input_layer, conv1_w, strides=[1, 1, 1, 1], padding='SAME')
    # print conv1
    relu1 = tf.nn.relu(conv1)
    # print relu1
    # pool1 = tf.nn.max_pool(relu1, ksize=[1, height, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    # print poo1

    # Flatten tensor into a batch of vectors
    pool2_flat = tf.reshape(relu1, [-1, height * embedding_dim])
    # print pool2_flat

    # dense layer
    weight1 = tf.get_variable('weight1', [height * embedding_dim, 1024], initializer=initializer,
                              regularizer=tf.contrib.layers.l2_regularizer(0.01))
    bias1 = tf.get_variable('bias3', [1024], initializer=tf.constant_initializer(0.0))
    hidden1 = tf.nn.relu(tf.matmul(pool2_flat, weight1) + bias1)
    # print hidden1

    # if task == 'train':
    #     # Add dropout operation; 0.5 probability that element will be kept
    #     hidden1 = tf.nn.dropout(hidden1, 0.5)
    #     # print hidden1

    # Output Layer
    # smaller output dimension will cause a lager gen_cost
    weight2 = tf.get_variable('weight2', [1024, embedding_dim], initializer=initializer,
                              regularizer=tf.contrib.layers.l2_regularizer(0.01))
    bias2 = tf.get_variable('bias2', [embedding_dim], initializer=tf.constant_initializer(0.0))
    output = tf.nn.sigmoid(tf.matmul(hidden1, weight2) + bias2)
    # print output

    return output
