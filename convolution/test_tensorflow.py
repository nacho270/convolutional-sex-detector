import tensorflow as tf


def run_tf_exampe() -> str:
    hello = tf.constant('Hello, TensorFlow!')
    sess = tf.Session()
    return sess.run(hello)