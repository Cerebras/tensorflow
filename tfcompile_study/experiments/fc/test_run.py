import tensorflow as tf
import sys
from utils import run
import pytest

HIDDEN_LAYERS = 5
LABELS = 10
HIDDEN_SIZE = 256 if HIDDEN_LAYERS > 1 else LABELS
LR = 0.01
BATCH_SIZE = 64
FEAT_DIM = 784


def model_fn(features, labels, is_training=True):
    with tf.variable_scope("fc", use_resource=True):
        net = features
        for i in range(HIDDEN_LAYERS - 1):
            net = tf.keras.layers.Dense(units=HIDDEN_SIZE,
                                        name="hidden" + str(i),
                                        activation=tf.nn.relu)(net)
        logits = tf.keras.layers.Dense(units=LABELS)(inputs=net)
        cross_entropy = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(labels=labels,
                                                    logits=logits))
        train_step = tf.train.GradientDescentOptimizer(
            LR, name="final_node").minimize(cross_entropy)
        with tf.control_dependencies([train_step]):
            return tf.identity(cross_entropy, name="results")


@pytest.fixture()
def inputs():
    x = tf.placeholder(tf.float32, [BATCH_SIZE, FEAT_DIM], name='x')
    y = tf.placeholder(tf.float32, [BATCH_SIZE, LABELS], name='y')
    return x, y


def test_model(inputs):
    run(model_fn, inputs, "fc", is_training=True)
