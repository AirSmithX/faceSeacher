import tensorflow as tf
import numpy as np
import prettytensor as pt

class Baseline(object):
    coeffs = None

    def __init__(self , session=None):
        self.net = None
        self.session = session

    def create_net(self , shape):
        print(shape)
        self.x = tf.placeholder(tf.float32 , shape=[None , shape] , name="x")
        self.y = tf.placeholder(tf.float32 , shape=[None] , name="y")
        self.net = (pt.wrap(self.x).
                    fully_connected(64 , activation_fn=tf.nn.relu).
                    fully_connected(1))
        self.net = tf.reshape(self.net , (-1 ,))
        self.l2 = (self.net - self.y) * (self.net - self.y)
        self.train = tf.train.AdamOptimizer().minimize(self.l2)
        self.session.run(tf.initialize_all_variables())

    def _features(self, path):
        obs = path["observations"]
        l = len(path["rewards"])
        al = np.arange(l).reshape(-1, 1) / 10.0
        ret = np.concatenate([obs, al, np.ones((l, 1))], axis=1)
        return ret

    def fit(self , paths):
        featmat = np.concatenate([self._features(path) for path in paths])
        if self.net is None:
            self.create_net(featmat.shape[1])
        returns = np.concatenate([path["returns"] for path in paths])
        for _ in range(100):
            loss, _ = self.session.run([self.l2, self.train], {self.x: featmat , self.y: returns})

    def predict(self , path):
        if self.net is None:
            return np.zeros(len(path["rewards"]))
        else:
            ret = self.session.run(self.net , {self.x: self._features(path)})
            return np.reshape(ret , (ret.shape[0] ,))
