import tensorflow as tf
import sys
sys.path.insert(0, "../../.")
from utils import run

HIDDEN_LAYERS = 2
LABELS = 10
HIDDEN_SIZE = 256 if HIDDEN_LAYERS > 1 else LABELS
LR = 0.01
BATCH_SIZE = 64
FEAT_DIM = 784


def model_fn(features, labels):
    with tf.variable_scope("fc", use_resource=True):
        net = features
        updates = []
        for i in range(HIDDEN_LAYERS - 1):
            net = tf.keras.layers.Dense(units=HIDDEN_SIZE,
                                        name="hidden" + str(i),
                                        activation=tf.nn.relu)(net)
            m = tf.keras.layers.BatchNormalization()
            net = m(net, training=True)
            updates += m.updates
        logits = tf.keras.layers.Dense(units=LABELS)(net)
        cross_entropy = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(labels=labels,
                                                    logits=logits))
        train_step = tf.train.GradientDescentOptimizer(
            LR, name="final_node").minimize(cross_entropy)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        train_step = tf.group([train_step, update_ops])
        with tf.control_dependencies([train_step] + updates):
            return tf.identity(cross_entropy, name="results")


def input_fn():
    x = tf.placeholder(tf.float32, [BATCH_SIZE, FEAT_DIM], name='x')
    y = tf.placeholder(tf.float32, [BATCH_SIZE, LABELS], name='y')
    return x, y


if __name__ == "__main__":
    run(model_fn, input_fn, "fc_keras_batchnorm")
