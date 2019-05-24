#!/usr/bin/env python3
import tensorflow as tf
from utils import run
import pytest
from tensorflow.keras.layers import LSTMCell, Dense

TGT_VOCAB_SIZE = 5000
SEQ_LENS = [12, 5, 10]

BATCH_SIZE = len(SEQ_LENS)
MAX_SEQ_LEN = max(SEQ_LENS)

LSTM_UNITS = 1000
LEARNING_RATE = 0.7


def model_fn(inputs, labels, is_training=True):
    sequence_lengths = tf.constant(SEQ_LENS)
    outputs, _ = tf.nn.dynamic_rnn(LSTMCell(LSTM_UNITS),
                                   inputs,
                                   sequence_length=sequence_lengths,
                                   dtype=tf.float32)
    logits = Dense(TGT_VOCAB_SIZE)(outputs)
    loss = tf.reduce_mean(
        tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels,
                                                       logits=logits))
    train_op = tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(
        loss, global_step=tf.train.get_global_step())
    with tf.control_dependencies([train_op]):
        return tf.identity(loss)


@pytest.fixture()
def inputs():
    inputs_holder = tf.placeholder(tf.float32,
                                   shape=[BATCH_SIZE, MAX_SEQ_LEN, 1])
    labels_holder = tf.placeholder(tf.int32, shape=[BATCH_SIZE, MAX_SEQ_LEN])
    return inputs_holder, labels_holder


def test_model(inputs):
    run(model_fn, inputs, "rnn_seq_length", is_training=True)
