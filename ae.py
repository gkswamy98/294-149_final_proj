import tensorflow as tf
import time

from model import Model
from tqdm import tqdm

class AutoEncoder(Model):

    def construct_model(self):
        x_ch = tf.expand_dims(self.x, 3)
        e1 = tf.layers.conv2d(x_ch, 3, 3, strides=2, padding='same', activation=tf.tanh,
                              name='e1')
        e2 = tf.layers.conv2d(e1, 6, 3, strides=2, padding='same', activation=tf.tanh,
                              name='e2')
        e2_flat = tf.reshape(e2, [-1, 6*7*7])

        self.latent(e2_flat)

        d3_flat = tf.layers.dense(self.hid, 6*7*7, activation=tf.tanh, name='d3')
        d3 = tf.reshape(d3_flat, [-1, 7, 7, 6])
        d4 = tf.layers.conv2d_transpose(d3, 3, 3, strides=2, padding='same',
                                        activation=tf.tanh, name='d4')
        o_ch = tf.layers.conv2d_transpose(d4, 1, 3, strides=2, padding='same', name='out')
        self.o = tf.squeeze(o_ch, 3)

        x_flat = tf.reshape(self.x, [-1, 28*28])
        o_flat = tf.reshape(self.o, [-1, 28*28])

        self.ae_loss = self.autoencoder_loss(x_flat, o_flat)
        self.ae_optim = tf.train.AdamOptimizer(self.params['learning_rate'])
        self.ae_step = self.ae_optim.minimize(self.ae_loss, var_list=tf.get_collection(
                                              tf.GraphKeys.GLOBAL_VARIABLES, scope='model'))

    def latent(self, l_in):
        self.hid = tf.layers.dense(l_in, 128, name='hid')

    def autoencoder_loss(self, label, logit):
        return tf.reduce_mean(tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(
                              labels=label, logits=logit), axis=1))

    def construct_classifier(self):
        for i in range(self.params['num_hid']):
            self.hid = tf.layers.dense(self.hid, 256, activation=tf.nn.relu, name='fc_%d' % i)

        super(AutoEncoder, self).construct_classifier()

    def classifier_variables(self):
        model_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='model')
        global_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
        return [var for var in global_vars if var not in model_vars]

    def train(self, epochs, to_filter):
        
        tqdm.write('Training autoencoder...')
        for epoch in range(1, epochs+1):
            tqdm.write('Training epoch %d' % epoch)
            time.sleep(0.2)
            batch_gen = self.batcher.batch_generator(self.params['batch_size'], to_filter)
            ae_loss = float('inf')
            while True:
                try:
                    img, _ = next(batch_gen)
                    feed_dict = {self.x:img}
                    ae_loss, _ = self.sess.run([self.ae_loss, self.ae_step], feed_dict)
                except StopIteration:
                    tqdm.write('AE epoch %d completed; loss is %f' % (epoch, round(ae_loss, 3)))
                    break

        tqdm.write('Autoencoder training completed. Training classifier...')
        super(AutoEncoder, self).train(epochs, to_filter)

class VAE(AutoEncoder):

    def latent(self, l_in):
        latent_dim = 128

        self.mean = tf.layers.dense(l_in, latent_dim, name='mean')
        self.logvar = tf.layers.dense(l_in, latent_dim, name='logvar')
        normal = tf.random_normal(shape=[tf.shape(l_in)[0],latent_dim])

        self.hid = tf.add(self.mean, tf.exp(self.logvar / 2) * normal, name='hid')

    def autoencoder_loss(self, label, logit):
        r_loss = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels=label, logits=logit),
                               axis=1)
        kl_loss = 0.5 * tf.reduce_sum(tf.math.exp(self.logvar) + self.mean**2 - self.logvar - 1,
                                      axis=1)
        return tf.reduce_mean(r_loss + kl_loss)
