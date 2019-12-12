#!/usr/bin/env python3
"""
This provides a simple Conv-Pool-Conv-FC tensorflow model
for testing xla extraction
"""

import os

#os.environ["XLA_FLAGS"] = "--xla_dump_hlo_as_text --xla_dump_to=./hlo"

import tensorflow as tf
import logging
import numpy as np
import shutil
from tensorflow.python import keras
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.contrib.compiler.xla import compile

_WITH_SUMMARIES = True

def model_fn(features, labels, mode=tf.estimator.ModeKeys.TRAIN, params=None):
    ''' This function is the input to Estimator constructor.
    More generally it is a python function that returns a computational graph
    given some set of inputs
    '''
    num_classes = 10

    data_format = "channels_first"
    with tf.variable_scope("eg_model", use_resource=True):
        with tf.device("/job:localhost/replica:0/task:0/device:XLA_CPU:0"):
            conv1 = keras.layers.Conv2D(
                filters=4,
                kernel_size=[3, 3],
                padding="same",
                activation=tf.nn.relu,
                data_format=data_format)(features)
            pool1 = keras.layers.MaxPooling2D(
                pool_size=(2, 2),
                data_format=data_format)(conv1)

            if _WITH_SUMMARIES:
                #with tf.device("/job:localhost/replica:0/task:0/device:XLA_GPU:0"):
                #with tf.device('/job:bar/task:0/device:cpu:0'):
                    tf.summary.scalar('summary_pool1_max', tf.math.reduce_max(pool1, name="my_reduce_max"))

            conv2 = keras.layers.Conv2D(
                filters=4,
                kernel_size=[3, 3],
                padding="same",
                activation=tf.nn.relu,
                data_format=data_format)(pool1)

            if _WITH_SUMMARIES:
                tf.summary.image('summary_conv2_image', conv2)

            flat = keras.layers.Flatten(
                data_format=data_format)(conv2)
            logits = keras.layers.Dense(
                units=num_classes,
                use_bias=False)(flat)

            labels = tf.one_hot(labels, depth=num_classes)
            loss = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits_v2(
                    labels=labels, logits=logits), name='loss')

            if _WITH_SUMMARIES:
                tf.summary.scalar('summary_mean_loss', tf.math.reduce_mean(loss))

            learning_rate = 0.1
            train_op = \
                tf.train.GradientDescentOptimizer(
                    learning_rate, name="train_step").minimize(
                    loss, global_step=tf.train.get_global_step())

            with ops.control_dependencies([train_op]):
                return array_ops.identity(loss, name=loss.op.name)

xshape, yshape = [16, 3, 32, 32], [16, 1]

# Constant input version
#x = tf.constant(20, tf.float32, shape=xshape)
#y = tf.constant(1, tf.int32, shape=yshape)

# Placeholder input version
x = tf.placeholder(tf.float32, shape=xshape)
y = tf.placeholder(tf.int32, shape=yshape)

def generic_compile(model_fn, inputs):
    placeholder_inputs = [
        tf.placeholder(i.dtype, shape=i.shape, name=i.op.name) for i in inputs]
    return compile(model_fn, inputs=placeholder_inputs)

(loss,) = generic_compile(model_fn, inputs=[x, y])

from tensorflow.tools.xla_extract import XlaExtract

hlo_mod = XlaExtract(loss)

hlo_mod_string = hlo_mod.SerializeToString()

with open("xla_out.proto", 'w') as f:
    f.write(str(hlo_mod))

with open("xla_out.bin", 'w') as f:
    f.write(str(hlo_mod_string))

def _draw_graph(logdir):
    with tf.Session() as sess:
        writer = tf.summary.FileWriter(logdir, sess.graph)

# https://github.com/tensorflow/tensorflow/commit/39587aaeb7760c6c963d8668bd6f614360e914f8#diff-4b1165fb6148e99a9fb6fed2dd8979a6R189

# XLA dump: https://groups.google.com/forum/#!topic/xla-dev/nMzNZfk-Jhw

TB_LOGDIR = "/tmp/tblogdir"
if os.path.exists(TB_LOGDIR):
    shutil.rmtree(TB_LOGDIR)
    os.mkdir(TB_LOGDIR, mode=0o777)

_draw_graph(TB_LOGDIR)


