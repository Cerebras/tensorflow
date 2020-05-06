#!/usr/bin/env python3
"""
This provides a simple Conv-Pool-Conv-FC tensorflow model
for testing xla extraction
"""

import tensorflow as tf
import logging
import numpy as np
from tensorflow.python import keras
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.compiler.xla import xla


def model_fn(features, labels, mode=tf.estimator.ModeKeys.TRAIN, params=None):
    ''' This function is the input to Estimator constructor.
    More generally it is a python function that returns a computational graph
    given some set of inputs
    '''
    num_classes = 10

    data_format = "channels_first"
    #jit_scope = tf.python.compiler.jit.experimental_jit_scope
    with tf.variable_scope("eg_model", use_resource=True):
        flat = keras.layers.Flatten(
            data_format=data_format)(features)
        logits = keras.layers.Dense(
            units=num_classes,
            use_bias=True)(flat)

        labels = tf.one_hot(labels, depth=num_classes)
        loss = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits_v2(
                labels=labels, logits=logits), name='loss')

        learning_rate = 0.1
        train_op = \
            tf.train.GradientDescentOptimizer(
                learning_rate, name="train_step").minimize(
                loss, global_step=tf.train.get_global_step())

        with ops.control_dependencies([train_op]):
            return array_ops.identity(loss, name=loss.op.name)

xshape, yshape = [128, 1, 28, 28], [128, 1]

# Constant input version
#x = tf.constant(20, tf.float32, shape=xshape)
#y = tf.constant(1, tf.int32, shape=yshape)

# Placeholder input version
x = tf.placeholder(tf.float32, shape=xshape)
y = tf.placeholder(tf.int32, shape=yshape)

def generic_compile(model_fn, inputs):
    placeholder_inputs = [
        tf.placeholder(i.dtype, shape=i.shape, name=i.op.name) for i in inputs]
    return xla.compile(model_fn, inputs=placeholder_inputs)

(loss,) = generic_compile(model_fn, inputs=[x, y])

from tensorflow.tools.xla_extract import XlaExtract
print(loss)
hlo_mod = XlaExtract(loss)

with open("xla_out.pbtxt", 'w') as f:
    f.write(str(hlo_mod))
