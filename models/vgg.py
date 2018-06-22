import collections
import chainer.functions as F
import chainer.links as L

from lib import utils
from GAIN import GAIN

class VGG16_GAIN(GAIN):

	def __init__(self):
		super(VGG16_GAIN, self).__init__()
		with self.init_scope():
			self.conv1_1 = L.Convolution2D(3, 64, 3, 1, 1)
			self.conv1_2 = L.Convolution2D(64, 64, 3, 1, 1)
			self.conv2_1 = L.Convolution2D(64, 128, 3, 1, 1)
			self.conv2_2 = L.Convolution2D(128, 128, 3, 1, 1)
			self.conv3_1 = L.Convolution2D(128, 256, 3, 1, 1)
			self.conv3_2 = L.Convolution2D(256, 256, 3, 1, 1)
			self.conv3_3 = L.Convolution2D(256, 256, 3, 1, 1)
			self.conv4_1 = L.Convolution2D(256, 512, 3, 1, 1)
			self.conv4_2 = L.Convolution2D(512, 512, 3, 1, 1)
			self.conv4_3 = L.Convolution2D(512, 512, 3, 1, 1)
			self.conv5_1 = L.Convolution2D(512, 512, 3, 1, 1)
			self.conv5_2 = L.Convolution2D(512, 512, 3, 1, 1)
			self.conv5_3 = L.Convolution2D(512, 512, 3, 1, 1)
			self.fc6 = L.Linear(512 * 7 * 7, 4096)
			self.fc7 = L.Linear(4096, 4096)
			self.fc8 = L.Linear(4096, 1000)

		utils._retrieve(
			'VGG_ILSVRC_16_layers.npz',
			'http://www.robots.ox.ac.uk/%7Evgg/software/very_deep/'
			'caffe/VGG_ILSVRC_16_layers.caffemodel',
			self)

		self.size = 224
		self.functions = collections.OrderedDict([
			('conv1_1', [self.conv1_1, F.relu]),
			('conv1_2', [self.conv1_2, F.relu]),
			('pool1', [_max_pooling_2d]),
			('conv2_1', [self.conv2_1, F.relu]),
			('conv2_2', [self.conv2_2, F.relu]),
			('pool2', [_max_pooling_2d]),
			('conv3_1', [self.conv3_1, F.relu]),
			('conv3_2', [self.conv3_2, F.relu]),
			('conv3_3', [self.conv3_3, F.relu]),
			('pool3', [_max_pooling_2d]),
			('conv4_1', [self.conv4_1, F.relu]),
			('conv4_2', [self.conv4_2, F.relu]),
			('conv4_3', [self.conv4_3, F.relu]),
			('pool4', [_max_pooling_2d]),
			('conv5_1', [self.conv5_1, F.relu]),
			('conv5_2', [self.conv5_2, F.relu]),
			('conv5_3', [self.conv5_3, F.relu]),
			('pool5', [_max_pooling_2d]),
			('fc6', [self.fc6, F.relu, F.dropout]),
			('fc7', [self.fc7, F.relu, F.dropout]),
			('fc8', [self.fc8]),
			('prob', [F.softmax]),
		])

def _max_pooling_2d(x):
	return F.max_pooling_2d(x, ksize=2)