import numpy as np
import random

from mnist import MNIST
from tqdm import tqdm

class Batcher:

    def __init__(self, data_path):
        data = MNIST(data_path)
        train_img, train_lbl = data.load_training()
        self.test_img, self.test_lbl = data.load_testing()
        self.test_img = np.array(self.test_img).reshape(len(self.test_img),28,28)/256
        self.test_lbl = np.array(self.test_lbl)
        self.train_samples = []
        for i in range(len(train_lbl)):
            self.train_samples.append((np.array(train_img[i]).reshape(28,28)/256, train_lbl[i]))

    def batch_generator(self, batch_size, to_filter):
        filtered = list(filter(lambda x: x[1] not in to_filter, self.train_samples))
        random.shuffle(filtered)
        for i in tqdm(range(0, len(filtered), batch_size)):
            imgs = np.array([x[0] for x in filtered[i:i+batch_size]])
            lbls = np.array([x[1] for x in filtered[i:i+batch_size]])
            yield imgs, lbls
