import tensorflow as tf

a = [[0, 0, 1], [1, 0, 0]]
b = [[1, 0, 0], [1, 0, 0]]
with tf.Session() as sess:
    correct_preds = tf.equal(tf.arg_max(a, 1), tf.arg_max(b, 1))
    accuracy = tf.reduce_sum(tf.cast(correct_preds, tf.float32))
    print(sess.run([correct_preds, accuracy]))
