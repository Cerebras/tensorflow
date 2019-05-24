import tensorflow as tf
from utils import run
import pytest

SEQ_LENGTH = 100
BATCH_SIZE = 16
NUM_CLASSES = 10
STATE_SIZE = 32


def model_fn(features, labels, is_training=True):
    with tf.variable_scope("keras_rnn", use_resource=True,
                           reuse=tf.AUTO_REUSE):
        rnn_cell = tf.keras.layers.SimpleRNNCell(STATE_SIZE)
        outputs = tf.keras.layers.RNN(cell=rnn_cell)(features)
        logits = tf.keras.layers.Dense(units=NUM_CLASSES)(outputs)
        loss = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(logits=logits,
                                                    labels=labels))

        train_op = tf.train.GradientDescentOptimizer(
            learning_rate=0.15,
            name="train_op").minimize(loss,
                                      global_step=tf.train.get_global_step())

        with tf.control_dependencies([train_op]):
            return tf.identity(loss)


@pytest.fixture()
def inputs():
    xshape, yshape = [BATCH_SIZE, SEQ_LENGTH, 1], [BATCH_SIZE, NUM_CLASSES]
    x = tf.placeholder(tf.float32, shape=xshape, name='x')
    y = tf.placeholder(tf.float32, shape=yshape, name='y')
    return x, y


def test_model(inputs):
    run(model_fn, inputs, "keras_rnn", is_training=True)
