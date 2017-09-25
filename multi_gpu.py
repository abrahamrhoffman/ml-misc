import tensorflow as tf

z = []
for GPU in ['/gpu:0', '/gpu:1', '/gpu:2']:
  with tf.device(GPU):
    a = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3])
    b = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[3, 2])
    c = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3])
    z.append(tf.matmul(a, b) + c)
with tf.device('/cpu:0'):
  sum = tf.add_n(z)
# Creates a session with log_device_placement set to True.
sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
# Runs the op.
print(sess.run(sum))
