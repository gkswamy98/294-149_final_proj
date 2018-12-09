import tensorflow as tf

from model import Model

class Convolutional(Model):

    def construct_model(self):
        preconv = tf.expand_dims(self.x, axis=3)

        for i in range(self.params['num_conv']):
            preconv = tf.layers.conv2d(preconv, 3, 3, padding='same', activation=tf.tanh,
                                       name='conv_%d' % i)

        c1 = tf.layers.conv2d(preconv, 6, 5, padding='same', activation=tf.tanh,
                              name='c1')
        s2 = tf.layers.max_pooling2d(c1, 2, 2, name='s2')
        c3 = tf.layers.conv2d(s2, 16, 5, activation=tf.tanh, name='c3')
        s4 = tf.layers.max_pooling2d(c3, 2, 2, name='s4')
        s4_flat = tf.reshape(s4, [-1, 5*5*16])
        c5 = tf.layers.dense(s4_flat, 120, activation=tf.tanh, name='c5')
        self.hid = tf.layers.dense(c5, 84, activation=tf.tanh, name='f6')
