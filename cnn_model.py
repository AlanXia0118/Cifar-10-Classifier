import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import math

'''
block x has the shape of [N, H, W]

'''
# set dropout rate
dropout_rate = 0.5


# batch_norm layer for later utilities
def batch_norm_layer(x, train_phase, scope_bn):
    # reshape the original data block to [N(#feature maps), L(#unrolled pixels)]
    shape = x.get_shape().as_list()
    x_unrolled = tf.reshape(x, [-1, shape[1] * shape[2] * shape[3]])
    with tf.variable_scope(scope_bn):
        beta = tf.Variable(tf.constant(0.0, shape=[x_unrolled.shape[-1]]), name='beta', trainable=True)
        gamma = tf.Variable(tf.constant(1.0, shape=[x_unrolled.shape[-1]]), name='gamma', trainable=True)
        batch_mean, batch_var = tf.nn.moments(x_unrolled, axes=[0], name='moments')
        # create an ExpMovingAver object
        ema = tf.train.ExponentialMovingAverage(decay=0.5)

        def mean_var_with_update():
            # update the shadow variable for batch_mean, batch_var
            ema_apply_op = ema.apply([batch_mean, batch_var])
            with tf.control_dependencies([ema_apply_op]):
                return tf.identity(batch_mean), tf.identity(batch_var)

        # execute true_fn()=mean_var_with_update if training, to update shadow variable as empirical mean / var
        # otherwise return f()=lambda : ( , ) as empirical mean / var for predicting
        mean, var = tf.cond(train_phase, mean_var_with_update,
                            lambda: (ema.average(batch_mean), ema.average(batch_var)))

        normed = tf.nn.batch_normalization(x_unrolled, mean, var, beta, gamma, 1e-3)
        normed = tf.reshape(normed, tf.shape(x))
    return normed


x = tf.placeholder(tf.float32, [None, 32, 32, 3])
y = tf.placeholder(tf.int64, [None])
is_training = tf.placeholder(tf.bool)


def cnn_model(X, y, is_training):
    # Given Batch of image has the shape of [N = batch, H, W, C]

    # regularizer = tf.contrib.layers.l2_regularizer(scale=0.4)

    # Set train_phase for batch_normalization
    train_phase = tf.convert_to_tensor(is_training)

    # Layer Conv1
    # conv1 has the shape of [N, 32, 32, 64]
    # param = 27*64 + 64 = 1792
    Wconv1 = tf.get_variable("Wconv1", shape=[3, 3, 3, 64])
    bconv1 = tf.get_variable("bconv1", shape=[64])
    conv1 = tf.nn.conv2d(X, Wconv1, strides=[1, 1, 1, 1], padding='SAME') + bconv1

    # Layer BN1
    # ouput sha[e = [N, 32, 32, 64]
    # params = 32*32*64*2 = 131072
    scope_bn1 = 'BN1'
    batch_norm1 = batch_norm_layer(conv1, train_phase, scope_bn1)
    relu1 = tf.nn.relu(batch_norm1, 'relu1')


    # Layer Conv2
    # Conv2 shape = [N, 32, 32, 64], np.ndarray, dtype = float32
    # params = 9*64*64 + 64 = 36928
    Wconv2 = tf.get_variable("Wconv2", shape=[3, 3, 64, 64])
    bconv2 = tf.get_variable("bconv2", shape=[64])
    conv2 = tf.nn.conv2d(relu1, Wconv2, strides=[1, 1, 1, 1], padding='SAME') + bconv2

    # Layer Max_pooling2
    # output shape = [N, 16, 16, 64]
    max_pooling2 = tf.nn.max_pool(conv2, [1, 2, 2, 1], [1, 2, 2, 1], padding='VALID')
    max_pooling2 = tf.cast(max_pooling2, tf.float32)

    # Layer Batch_normalization2
    # output shape = [N, 16, 16, 64]
    # train_phase should be tf.bool
    # params = 16*16*64*2 = 32768
    scope_bn2 = 'BN2'
    batch_norm2 = batch_norm_layer(max_pooling2, train_phase, scope_bn2)
    relu2 = tf.nn.relu(batch_norm2, 'relu2')
    # dropout
    relu2_dropout = tf.layers.dropout(relu2, dropout_rate, training=is_training)


    # Layer Conv3
    # output shape = [N, 8, 8, 128]
    # params = 9*64*128 + 128 = 73856
    Wconv3 = tf.get_variable("Wconv3", shape=[3, 3, 64, 128])
    bconv3 = tf.get_variable("bconv3", shape=[128])
    conv3 = tf.nn.conv2d(relu2_dropout, Wconv3, strides=[1, 2, 2, 1], padding='SAME') + bconv3

    # Layer Batch_normalization3
    # output shape = [N, 8, 8, 128]
    # train_phase should be tf.bool
    # params = 8*8*128*2 = 8192
    scope_bn3 = 'BN3'
    batch_norm3 = batch_norm_layer(conv3, train_phase, scope_bn3)
    relu3 = tf.nn.relu(batch_norm3, 'relu3')
    # dropout
    relu3_dropout = tf.layers.dropout(relu3, dropout_rate, training=is_training)


    # Layer Conv4
    # output shape = [N, 4, 4, 64]
    # params = 3*3*128*64 + 64 = 73792
    Wconv4 = tf.get_variable("Wconv4", shape=[3, 3, 128, 64])
    bconv4 = tf.get_variable("bconv4", shape=[64])
    conv4 = tf.nn.conv2d(relu3_dropout, Wconv4, strides=[1, 2, 2, 1], padding='SAME') + bconv4

    # Layer Batch_normalization4
    # output shape = [N, 4, 4, 64]
    # train_phase should be tf.bool
    # params = 2048
    scope_bn4 = 'BN4'
    batch_norm4 = batch_norm_layer(conv4, train_phase, scope_bn4)
    relu4 = tf.nn.relu(batch_norm4, 'relu4')


    # Layer FC5
    # Fully conneted layer to compute scores
    # params = 10250
    W5 = tf.get_variable("W5", shape=[1024, 10])
    b5 = tf.get_variable("b5", shape=[10])
    relu4_flat = tf.reshape(relu4, [-1, 1024])

    # Dropout
    fc5_dropout = tf.layers.dropout(relu4_flat, dropout_rate, training=is_training)

    y_out = tf.matmul(fc5_dropout, W5) + b5

    # params_sum = 370698
    # y_out has the shape of [N, 10], is ndarray of dtype = float32

    return y_out
