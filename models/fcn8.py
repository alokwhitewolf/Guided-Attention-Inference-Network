import os.path as osp
import collections

import chainer
import chainer.functions as F
import chainer.links as L
import numpy as np
from GAIN import GAIN

import fcn
from fcn import data
from fcn import initializers


class FCN8s(GAIN):

	pretrained_model = osp.expanduser(
		'~/data/models/chainer/fcn8s_from_caffe.npz')

	def __init__(self, n_class=21):
		self.n_class = n_class
		kwargs = {
			'initialW': chainer.initializers.Zero(),
			'initial_bias': chainer.initializers.Zero(),
		}
		super(FCN8s, self).__init__()
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

			self.fc6 = L.Convolution2D(512, 4096, 7, 1, 0)
			self.fc7 = L.Convolution2D(4096, 4096, 1, 1, 0)
			self.score_fr = L.Convolution2D(4096, n_class, 1, 1, 0)

			self.fc6_cl = L.Linear(512, 4096)
			self.fc7_cl = L.Linear(4096, 4096)
			self.score_cl = L.Linear(4096, n_class-1) # Disregard 0 class for classification


			self.upscore2 = L.Deconvolution2D(
				n_class, n_class, 4, 2, 0, nobias=True,
				initialW=initializers.UpsamplingDeconvWeight())
			self.upscore8 = L.Deconvolution2D(
				n_class, n_class, 16, 8, 0, nobias=True,
				initialW=initializers.UpsamplingDeconvWeight())

			self.score_pool3 = L.Convolution2D(256, n_class, 1, 1, 0)
			self.score_pool4 = L.Convolution2D(512, n_class, 1, 1, 0)
			self.upscore_pool4 = L.Deconvolution2D(
				n_class, n_class, 4, 2, 0, nobias=True,
				initialW=initializers.UpsamplingDeconvWeight())

		self.GAIN_functions = collections.OrderedDict([
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

			('conv3_1', [self.conv5_1, F.relu]),
			('conv3_2', [self.conv5_2, F.relu]),
			('conv3_3', [self.conv5_3, F.relu]),
			('pool5', [_max_pooling_2d]),

			('avg_pool', [_average_pooling_2d]),

			('fc6_cl', [self.fc6_cl, F.relu]),
			('fc7_cl', [self.fc7_cl, F.relu]),
			('prob', [self.score_cl, F.sigmoid])

		])
		self.final_conv_layer = 'avg_pool'
		self.grad_target_layer = 'prob'
		self.freezed_layers = ['fc6_cl', 'fc7_cl', 'score_cl']

	def segment(self, x, t=None):
		# conv1
		self.conv1_1.pad = (100, 100)
		h = F.relu(self.conv1_1(x))
		conv1_1 = h
		h = F.relu(self.conv1_2(conv1_1))
		conv1_2 = h
		h = _max_pooling_2d(conv1_2)
		pool1 = h  # 1/2

		# conv2
		h = F.relu(self.conv2_1(pool1))
		conv2_1 = h
		h = F.relu(self.conv2_2(conv2_1))
		conv2_2 = h
		h = _max_pooling_2d(conv2_2)
		pool2 = h  # 1/4

		# conv3
		h = F.relu(self.conv3_1(pool2))
		conv3_1 = h
		h = F.relu(self.conv3_2(conv3_1))
		conv3_2 = h
		h = F.relu(self.conv3_3(conv3_2))
		conv3_3 = h
		h = _max_pooling_2d(conv3_3)
		pool3 = h  # 1/8

		# conv4
		h = F.relu(self.conv4_1(pool3))
		h = F.relu(self.conv4_2(h))
		h = F.relu(self.conv4_3(h))
		h = _max_pooling_2d(h)
		pool4 = h  # 1/16

		# conv5
		h = F.relu(self.conv5_1(pool4))
		h = F.relu(self.conv5_2(h))
		h = F.relu(self.conv5_3(h))
		h = _max_pooling_2d(h)
		pool5 = h  # 1/32

		# fc6
		h = F.relu(self.fc6(pool5))
		h = F.dropout(h, ratio=.5)
		fc6 = h  # 1/32

		# fc7
		h = F.relu(self.fc7(fc6))
		h = F.dropout(h, ratio=.5)
		fc7 = h  # 1/32

		# score_fr
		h = self.score_fr(fc7)
		score_fr = h  # 1/32

		# score_pool3
		h = self.score_pool3(pool3)
		score_pool3 = h  # 1/8

		# score_pool4
		h = self.score_pool4(pool4)
		score_pool4 = h  # 1/16

		# upscore2
		h = self.upscore2(score_fr)
		upscore2 = h  # 1/16

		# score_pool4c
		h = score_pool4[:, :,
						5:5 + upscore2.shape[2],
						5:5 + upscore2.shape[3]]
		score_pool4c = h  # 1/16

		# fuse_pool4
		h = upscore2 + score_pool4c
		fuse_pool4 = h  # 1/16

		# upscore_pool4
		h = self.upscore_pool4(fuse_pool4)
		upscore_pool4 = h  # 1/8

		# score_pool4c
		h = score_pool3[:, :,
						9:9 + upscore_pool4.shape[2],
						9:9 + upscore_pool4.shape[3]]
		score_pool3c = h  # 1/8

		# fuse_pool3
		h = upscore_pool4 + score_pool3c
		fuse_pool3 = h  # 1/8

		# upscore8
		h = self.upscore8(fuse_pool3)
		upscore8 = h  # 1/1

		# score
		h = upscore8[:, :, 31:31 + x.shape[2], 31:31 + x.shape[3]]
		score = h  # 1/1
		self.score = score

		if t is None:
			assert not chainer.config.train
			return

		loss = F.softmax_cross_entropy(score, t, normalize=True)
		if np.isnan(float(loss.data)):
			raise ValueError('Loss is nan.')
		chainer.report({'loss': loss}, self)
		self.conv1_1.pad = (1, 1)
		return loss


	def classify(self, x, is_training=True):
		with chainer.no_backprop_mode():
			# convv1
			h = F.relu(self.conv1_1(x))
			h = F.relu(self.conv1_2(h))
			h = _max_pooling_2d(h)

			# conv2
			h = F.relu(self.conv2_1(h))
			h = F.relu(self.conv2_2(h))
			h = _max_pooling_2d(h)

			# conv3
			h = F.relu(self.conv3_1(h))
			h = F.relu(self.conv3_2(h))
			h = F.relu(self.conv3_3(h))
			h = _max_pooling_2d(h)

			# conv4
			h = F.relu(self.conv4_1(h))
			h = F.relu(self.conv4_2(h))
			h = F.relu(self.conv4_3(h))
			h = _max_pooling_2d(h)

			# conv5
			h = F.relu(self.conv5_1(h))
			h = F.relu(self.conv5_2(h))
			h = F.relu(self.conv5_3(h))
			h = _max_pooling_2d(h)
			h = _average_pooling_2d(h)

		with chainer.using_config('train',is_training):
			h = F.relu(F.dropout(self.fc6_cl(h), .5))
			h = F.relu(F.dropout(self.fc7_cl(h), .5))
			h = self.score_cl(h)

		return h


	def __call__(self, x, t=None):
		return self.segment(x, t)

	@classmethod
	def download(cls):
		return data.cached_download(
			url='https://drive.google.com/uc?id=0B9P1L--7Wd2vb0cxV0VhcG1Lb28',
			path=cls.pretrained_model,
			md5='256c2a8235c1c65e62e48d3284fbd384',
		)

	def predict(self, imgs):
		lbls = []
		for img in imgs:
			with chainer.no_backprop_mode(), chainer.using_config('train', False):
				x = self.xp.asarray(img[None])
				self.__call__(x)
				lbl = chainer.functions.argmax(self.score, axis=1)
			lbl = chainer.cuda.to_cpu(lbl.array[0])
			lbls.append(lbl)
		return lbls

def _max_pooling_2d(x):
	return F.max_pooling_2d(x, ksize=2, stride=2, pad=0)

def _average_pooling_2d(x):
	return F.average_pooling_2d(x, ksize=(x.shape[-2], x.shape[-1]))
