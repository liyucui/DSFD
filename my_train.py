# coding: utf-8

import os
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
# import Utils
import BatchData as dataSet

os.environ['CUDA_VISIBLE_DEVICES'] = "3"
data_dir = "./resized_WIDER"
MAX_ITERATION = int(18e4)
MODEL_SIZE = (640, 640)

tf.flags.DEFINE_integer("batch_size", "32", "batch size for training")
tf.flags.DEFINE_string("logs_dir", "logs/", "path tp logs directory")
tf.flags.DEFINE_float("learning_rate", "0.0001", "learning rate of adam optimizer")

RESTORE = False
dropout = 0.5
FLAGS = tf.flags.FLAGS

def trainDetectionModel(input_image):
    input_image /= 255.0
    with tf.variable_scope('trainClassifier'):
        with slim.arg_scope([slim.conv2d],
                            padding='SAME',
                            normalizer_fn=slim.batch_norm,
                            activation_fn=tf.nn.relu6,
                            weights_regularizer=slim.l2_regularizer(0.0005),
                            weights_initializer=slim.variance_scaling_initializer()):
            with slim.arg_scope([slim.batch_norm], scale=True, is_training=True):
                conv1 = slim.conv2d(input_image, 4, 3, stride=2)
                conv2 = slim.conv2d(conv1, 8, 3, stride=2)
                conv3 = slim.conv2d(conv2, 16, 3, stride=2)
                conv4 = slim.conv2d(conv3, 32, 3, stride=2)
                conv4 = tf.layers.flatten(conv4)
                conv5 = tf.layers.dense(inputs=conv4, units=512, activation=tf.nn.relu6)
                conv5 = tf.nn.dropout(conv5, dropout)
                classifier = tf.layers.dense(inputs=conv5, units=1, activation=tf.nn.sigmoid)
    return classifier

def valDetectionModel(input_image):
    input_image /= 255.0
    with tf.variable_scope('valClassifier'):
        with slim.arg_scope([slim.conv2d],
                            padding='SAME',
                            normalizer_fn=slim.batch_norm,
                            activation_fn=tf.nn.relu6,
                            weights_regularizer=slim.l2_regularizer(0.0005),
                            weights_initializer=slim.variance_scaling_initializer()):
            with slim.arg_scope([slim.batch_norm], scale=True, is_training=True):
                conv1 = slim.conv2d(input_image, 4, 3, stride=2)
                conv2 = slim.conv2d(conv1, 8, 3, stride=2)
                conv3 = slim.conv2d(conv2, 16, 3, stride=2)
                conv4 = slim.conv2d(conv3, 32, 3, stride=2)
                conv4 = tf.layers.flatten(conv4)
                conv5 = tf.layers.dense(inputs=conv4, units=512, activation=tf.nn.relu6)
                classifier = tf.layers.dense(inputs=conv5, units=1, activation=tf.nn.relu6)
    return classifier

def cal_cross_entropy_loss():



def main(argv=None):
    with tf.variable_scope('input'):
        image = tf.placeholder(
            tf.float32, shape=[None, MODEL_SIZE[0], MODEL_SIZE[1], 3], name="input_img")
        box = tf.placeholder(
            tf.float32, shape=[None, 4], name="input_box")
        labels = tf.placeholder(
            tf.float32, shape=[None, 1], name='input_label')

    train_detection = trainDetectionModel(image)
    val_detection = valDetectionModel(image)
    train_loss = cal_cross_entropy_loss(train_detection, label, useRegularization=True)
    train_op = tf.train.AdamOptimizer(FLAGS.learning_rate).minimize(train_loss)
    val_loss = cal_cross_entropy_loss(val_detection, label, useRegularization=True)


    img_data = dataSet.BatchData(data_dir)

    Loss_min = 999
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    # train_summary_writer = tf.summary.FileWriter(FLAGS.logs_dir+'train', sess.graph)
    # val_summary_writer = tf.summary.FileWriter(FLAGS.logs_dir+'val', sess.graph)

    if RESTORE:
        saver0 = tf.train.Saver()
        saver0.restore(sess, "logs/model.ckpt-10000")

    saver = tf.train.Saver(var_list=tf.global_variables(), max_to_keep=20)
    writer = tf.summary.FileWriter("logs", tf.get_default_graph())
    writer.close()

    for itr in range(0, MAX_ITERATION):
        train_images, train_labels = img_data.train_next_batch(FLAGS.batch_size)
        feed_dict = {image: train_images, label: train_labels}
        sess.run(train_op, feed_dict=feed_dict)

        if itr % 50 == 0:
            train_batch_loss = sess.run(train_loss, feed_dict=feed_dict)
            if itr % 200 == 0:
                print("Step: %d, Train_loss: %g" % (itr, train_batch_loss), end='')
                val_images, val_labels = img_data.val_random_batch(64)
                feed_dict_dev = {image: val_images, label: val_labels}
                val_batch_loss = sess.run(val_loss, feed_dict=feed_dict_dev)
                if val_batch_loss < Loss_min:
                    Loss_min = val_batch_loss
                    saver.save(sess, FLAGS.logs_dir + "minLossModel.ckpt", itr)
                print(" Val_loss: %g" % val_batch_loss)
        if itr % 800 == 0:
            saver.save(sess, FLAGS.logs_dir + "model.ckpt", itr)
            FLAGS.learning_rate /= 2.0
            if FLAGS.learning_rate < 0.0000001:
                FLAGS.learning_rate = 0.0000001
                print("Learning_rate: %g" % FLAGS.learning_rate)

    sess.close()
    print('end')

if __name__ == "__main__":
    tf.app.run()































