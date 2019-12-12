#!/usr/bin/env python3
"""
This provides a simple Conv-Pool-Conv-FC tensorflow model
for testing xla extraction
"""

import os

#os.environ["TF_XLA_FLAGS"] = "--xla_dump_executions_to = xla_out.comp"
#os.environ["XLA_FLAGS"] = "--xla_dump_hlo_as_text --xla_dump_to=./hlo"
#os.environ["XLA_FLAGS"] = "--xla_dump_hlo_as_text --xla_dump_to=./hlo"

os.environ["XLA_FLAGS"] = "--xla_hlo_graph_path=./tmp_dot --xla_generate_hlo_graph=.*"
os.environ["XLA_LOG"] = "3"

import tensorflow as tf
import logging
import numpy as np
import shutil
import pathlib
import graphviz
import re
import pydotplus

from pydotplus import graphviz as pydot

from os import listdir
from os.path import isfile, join

from graphviz import Source
from tensorflow.python import keras
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.contrib.compiler.xla import compile
from tensorflow.python.client import timeline

_WITH_SUMMARIES = True

def model_fn(features, labels, mode=tf.estimator.ModeKeys.TRAIN, params=None):
    ''' This function is the input to Estimator constructor.
    More generally it is a python function that returns a computational graph
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

        if _WITH_SUMMARIES:
            with tf.device('/job:bar/task:0/device:cpu:0'):
                tf.summary.scalar('summary_pool1_max', tf.math.reduce_max(pool1))

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

def generic_compile(model_fn, inputs):
    placeholder_inputs = [
        tf.placeholder(i.dtype, shape=i.shape, name=i.op.name) for i in inputs]
    return compile(model_fn, inputs=placeholder_inputs)

def _standard_compile(model_fn, features, labels):

    loss = model_fn(features, labels)

    config = tf.ConfigProto()
    # Turns on XLA JIT compilation.
    jit_level = tf.OptimizerOptions.ON_1

    config.graph_options.optimizer_options.global_jit_level = jit_level
    run_metadata = tf.RunMetadata()
    sess = tf.Session(config=config)
    tf.global_variables_initializer().run(session=sess)

    # Train
    g = tf.Graph()
    print(dir(g))
    train_loops = 1
    for i in range(train_loops):
      batch_xs, batch_ys = np.ones(xshape), np.ones(yshape)

      # Create a timeline for the last loop and export to json to view with
      # chrome://tracing/.
      if i == train_loops - 1:
        sess.run(loss,
                 feed_dict={features: batch_xs,
                            labels: batch_ys},
                 options=tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE),
                 run_metadata=run_metadata)
        trace = timeline.Timeline(step_stats=run_metadata.step_stats)
        with open('timeline.ctf.json', 'w') as trace_file:
          trace_file.write(trace.generate_chrome_trace_format())
      else:
        sess.run(loss, feed_dict={features: batch_xs, labels: batch_ys})

    # def render(self, filename=None, directory=None, view=False, cleanup=False,
    #            format=None, renderer=None, formatter=None,
    #            quiet=False, quiet_view=False):


def _show_files(path, start=0, end=None):
    all_files = [f for f in listdir(path) if isfile(join(path, f))]
    dot_files = [f for f in all_files if pathlib.Path(join(path, f)).suffix == ".dot"]

    file_dict = {}
    for fname in dot_files:
        temp = re.findall(r'\d+', fname) 
        if temp:
            file_dict[int(temp[0])] = fname

    if not end:
        end = len(file_dict)
    assert end >= start
    for i in range(end - start):
        if i in file_dict:
            dotf = join(path, file_dict[i])

            graph = pydot.graph_from_dot_file(dotf)

            s = Source.from_file(dotf)
            s.view()
            string = input("Press Enter to continue... (q+Enter to quit)")
            if (string == "q"):
                break


def main():
    with tf.device("/job:localhost/replica:0/task:0/device:XLA_GPU:0"):
        # Placeholder input version
        x = tf.placeholder(tf.float32, shape=xshape)
        y = tf.placeholder(tf.int32, shape=yshape)
        #w = tf.get_variable("w", initializer=tf.ones(xshape), use_resource=True)

        #x = x * w  # In order to get global var init to compile

        dot_dir = "./tmp_dot"
        if os.path.exists(dot_dir):
            shutil.rmtree(dot_dir)
        os.mkdir(dot_dir)
        
        _standard_compile(model_fn, x, y)
        _show_files(dot_dir, 0, 5)

        (loss,) = generic_compile(model_fn, inputs=[x, y])

        from tensorflow.tools.xla_extract import XlaExtract

        # Dump XLA, set environment flag: XLA_FLAGS="--xla_hlo_graph_path=./ --xla_generate_hlo_graph=.*"

        #train_op

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

main()
