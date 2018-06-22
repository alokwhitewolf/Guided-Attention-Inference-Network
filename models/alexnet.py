import collections

import chainer
import chainer.functions as F
import chainer.links as L

from lib import utils


class Alex(chainer.Chain):

    def __init__(self):
        super(Alex, self).__init__()
        with self.init_scope():
            self.conv1 = L.Convolution2D(3, 96, 11, stride=4)
            self.conv2 = L.Convolution2D(None, 256, 5, pad=2)
            self.conv3 = L.Convolution2D(None, 384, 3, pad=1)
            self.conv4 = L.Convolution2D(None, 384, 3, pad=1)
            self.conv5 = L.Convolution2D(None, 256, 3, pad=1)
            self.fc6 = L.Linear(None, 4096)
            self.fc7 = L.Linear(None, 4096)
            self.fc8 = L.Linear(None, 1000)

        utils._retrieve(
            'bvlc_alexnet.npz',
            'http://dl.caffe.berkeleyvision.org/bvlc_alexnet.caffemodel',
            self)

        self.size = 227
        self.functions = collections.OrderedDict([
            ('conv1', [self.conv1, F.relu, F.local_response_normalization]),
            ('pool1', [_max_pooling_2d]),
            ('conv2', [self.conv2, F.relu, F.local_response_normalization]),
            ('pool2', [_max_pooling_2d]),
            ('conv3', [self.conv3, F.relu]),
            ('conv4', [self.conv4, F.relu]),
            ('conv5', [self.conv5, F.relu]),
            ('pool5', [_max_pooling_2d]),
            ('fc6', [self.fc6, F.relu, F.dropout]),
            ('fc7', [self.fc7, F.relu, F.dropout]),
            ('fc8', [self.fc8]),
            ('prob', [F.softmax]),
        ])


def _max_pooling_2d(x):
	return F.max_pooling_2d(x, ksize=3, stride=2)