import tensorflow as tf
from utils import run
import pytest

HIDDEN_LAYERS = 2
LABELS = 10
HIDDEN_SIZE = 256 if HIDDEN_LAYERS > 1 else LABELS
LR = 0.01
BATCH_SIZE = 64
HEIGHT = 28
WIDTH = 28
CHANNELS = 3


def model_fn(features, labels, is_training=True):
    with tf.variable_scope("conv_fc", use_resource=True):
        net = tf.keras.layers.Conv2D(filters=4,
                                     kernel_size=[2, 2],
                                     activation=tf.nn.relu)(features)
        net = tf.layers.flatten(net)
        for i in range(HIDDEN_LAYERS - 1):
            net = tf.keras.layers.Dense(units=HIDDEN_SIZE,
                                        name="hidden" + str(i),
                                        activation=tf.nn.relu)(net)
        logits = tf.keras.layers.Dense(units=LABELS)(net)
        cross_entropy = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(labels=labels,
                                                    logits=logits))
        train_step = tf.train.AdamOptimizer(
            LR, name="final_node").minimize(cross_entropy)
        with tf.control_dependencies([train_step]):
            return tf.identity(cross_entropy, name="results")


@pytest.fixture()
def inputs():
    x = tf.placeholder(tf.float32, [BATCH_SIZE, HEIGHT, WIDTH, CHANNELS],
                       name='x')
    y = tf.placeholder(tf.float32, [BATCH_SIZE, LABELS], name='y')
    return x, y


def test_model(inputs):
    run(model_fn, inputs, file_name="conv_fc_adam", is_training=True)
