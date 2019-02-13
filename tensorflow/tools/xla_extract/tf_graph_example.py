# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""
This provides a simple Conv-Pool-Conv-FC tensorflow model
for testing xla extraction
"""
from tensorflow.tools.xla_extract import XlaExtract
import tensorflow as tf
from tensorflow.python import keras
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.contrib.compiler import xla


def model_fn(features, labels):
  '''
  It is a python function that returns a computational graph
  given some set of inputs
  '''
  num_classes = 10

  data_format = "channels_first"
  with tf.variable_scope("eg_model", use_resource=True):
    conv1 = keras.layers.Conv2D(
        filters=4,
        kernel_size=[3, 3],
        padding="same",
        activation=tf.nn.relu,
        data_format=data_format)(features)
    pool1 = keras.layers.MaxPooling2D(
        pool_size=(2, 2),
        data_format=data_format)(conv1)
    conv2 = keras.layers.Conv2D(
        filters=4,
        kernel_size=[3, 3],
        padding="same",
        activation=tf.nn.relu,
        data_format=data_format)(pool1)

    flat = keras.layers.Flatten(
        data_format=data_format)(conv2)
    logits = keras.layers.Dense(
        units=num_classes,
        use_bias=False)(flat)

    labels = tf.one_hot(labels, depth=num_classes)
    ce_loss = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits_v2(
            labels=labels, logits=logits), name='loss')

    learning_rate = 0.1
    train_op = \
      tf.train.GradientDescentOptimizer(
          learning_rate, name="train_step").minimize(
              ce_loss, global_step=tf.train.get_global_step())

    with ops.control_dependencies([train_op]):
      return array_ops.identity(ce_loss, name=loss.op.name)

xshape, yshape = [16, 3, 32, 32], [16, 1]

# Placeholder input version
x = tf.placeholder(tf.float32, shape=xshape)
y = tf.placeholder(tf.int32, shape=yshape)


def generic_compile(model_fn, inputs):
  placeholder_inputs = [
      tf.placeholder(i.dtype, shape=i.shape, name=i.op.name) for i in inputs]
  return xla.compile(model_fn, inputs=placeholder_inputs)

(loss,) = generic_compile(model_fn, inputs=[x, y])

hlo_mod = XlaExtract(loss)

with open("xla_out.pbtxt", 'w') as f:
  f.write(str(hlo_mod))
