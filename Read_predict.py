import numpy as np
import tensorflow as tf
import cnn_model as cnn
import math
import timeit
import matplotlib.pyplot as plt
import cv2

# the model directory where ckpt file was saved
model_path = './checkpoints/model.ckpt'
# # input your image path
img_path = './test_images/horse.png'


# classes of datasets
predict_list = ['airplane',
                'automobile',
                'bird',
                'cat',
                'deer',
                'dog',
                'frog',
                'horse',
                'ship',
                'truck']


# pre-process your image
img_raw = cv2.imread(img_path)
# since cv2.imread() load picture as B, G, R
img_rgb = img_raw[:, :, (2, 1, 0)]
size = (32, 32)
img_to_show = cv2.resize(img_rgb, size, interpolation=cv2.INTER_AREA)
# cv2.imwrite('test.png', img, [int(cv2.IMWRITE_PNG_COMPRESSION), 0])
img = img_to_show.reshape((1, 32, 32, 3))


# clear old variables
tf.reset_default_graph()


# setup input (e.g. the data)
# the first dim is None, and get sets based on batch size fed in
x = tf.placeholder(tf.float32, [None, 32, 32, 3])
y = tf.placeholder(tf.int64, [None])
is_training = tf.placeholder(tf.bool)


# use the imported model cnn_model
y_out = cnn.cnn_model(x, y, is_training)
saver = tf.train.Saver()


with tf.Session() as sess:
    saver.restore(sess, model_path)
    output = sess.run(y_out, feed_dict={x: img,
                                        is_training: False})

    out = np.reshape(output, [10])
    print('Scores: \n', out)
    print('Prediction: ', predict_list[np.argmax(out)])
    # visualize the prediction
    plt.ion()
    plt.imshow(img_to_show)
    plt.text(2, 15, predict_list[np.argmax(out)], size=60, alpha=1, color='c')
    plt.pause(5)











