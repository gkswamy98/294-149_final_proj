import numpy as np
import tensorflow as tf
import time

from tqdm import tqdm

class Model:

    def __init__(self, sess, batcher, params):
        self.sess = sess
        self.batcher = batcher
        self.params = params

        with tf.variable_scope('placeholders'):
            self.construct_placeholders()
        with tf.variable_scope('model'):
            self.construct_model()
        with tf.variable_scope('classifier'):
            self.construct_classifier()

    def construct_placeholders(self):
        self.x = tf.placeholder(tf.float32, shape=[None, 28, 28], name='input')
        self.y = tf.placeholder(tf.int32, shape=[None], name='label')

    def construct_model(self):
        self.hid = tf.reshape(self.x, [-1, 28*28])

    def construct_classifier(self):
        self.logits = tf.layers.dense(self.hid, 10, name='logits')
        self.loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
                                   labels=self.y, logits=self.logits, name='loss'))
        self.optim = tf.train.AdamOptimizer(self.params['learning_rate'])
        self.step = self.optim.minimize(self.loss, var_list=self.classifier_variables())

    def classifier_variables(self):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)

    def initialize_variables(self):
        self.sess.run(tf.global_variables_initializer())

    def train(self, epochs, to_filter):
        for epoch in range(1, epochs+1):
            tqdm.write('Training epoch %d' % epoch)
            time.sleep(0.2)
            batch_gen = self.batcher.batch_generator(self.params['batch_size'], to_filter)
            loss = float('inf')
            while True:
                try:
                    img, lbl = next(batch_gen)
                    feed_dict = {self.x:img, self.y:lbl}
                    loss, _ = self.sess.run([self.loss, self.step], feed_dict)
                except StopIteration:
                    tqdm.write('Epoch %d completed; loss is %f' % (epoch, round(loss, 3)))
                    break

    def predict(self):
        tqdm.write('Running predictions...')
        logits = []
        time.sleep(0.2)
        for i in tqdm(range(0, self.batcher.test_img.shape[0], self.params['batch_size'])):
            feed_dict = {self.x:self.batcher.test_img[i:i+self.params['batch_size']],
                         self.y:self.batcher.test_lbl[i:i+self.params['batch_size']]}
            logits.append(self.sess.run(self.logits, feed_dict))
        logits = np.concatenate(logits, axis=0)
        preds = np.argmax(logits, axis=1)
        correct = np.sum(np.equal(preds, self.batcher.test_lbl))
        tqdm.write('Test accuracy: %f' % (correct/self.batcher.test_lbl.shape[0]))
        return logits
