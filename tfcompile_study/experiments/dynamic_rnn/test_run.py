import tensorflow as tf
from utils import run
import pytest

SEQ_LENGTH = 100
BATCH_SIZE = 16
NUM_CLASSES = 10
STATE_SIZE = 32


def model_fn(features, labels, is_training=True):
    with tf.variable_scope("dynamic_rnn",
                           use_resource=True,
                           reuse=tf.AUTO_REUSE):
        rnn_cell = tf.nn.rnn_cell.BasicRNNCell(STATE_SIZE,
                                               name="rnn_cell",
                                               reuse=tf.AUTO_REUSE,
                                               dtype=tf.float32)
        initial_state = rnn_cell.zero_state(BATCH_SIZE, dtype=tf.float32)
        outputs, final_state = tf.nn.dynamic_rnn(cell=rnn_cell,
                                                 inputs=features,
                                                 initial_state=initial_state,
                                                 scope="func",
                                                 dtype=tf.float32)

        logits = tf.keras.layers.Dense(units=NUM_CLASSES)(outputs)
        logit_loss = logits[:, -1, :]
        logit_loss = tf.reshape(logit_loss, (BATCH_SIZE, NUM_CLASSES))
        loss = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(logits=logit_loss,
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
    run(model_fn, inputs, "dynamic_rnn", is_training=True)
