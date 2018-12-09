import tensorflow as tf

from model import Model

class FullyConnected(Model):

    def construct_model(self):
        super(FullyConnected, self).construct_model()
        for i in range(self.params['num_hid']):
            self.hid = tf.layers.dense(self.hid, 256, activation=tf.nn.relu, name='fc_%d' % i)
