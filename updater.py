from chainer.training import StandardUpdater
from chainer import Variable
from chainer import report
from chainer import functions as F
from chainer.backends.cuda import get_array_module
import numpy as np


class VOC_ClassificationUpdater(StandardUpdater):
	def __init__(self, iterator, optimizer, no_of_classes=21, device=-1):
		super(VOC_ClassificationUpdater, self).__init__(iterator, optimizer)
		# self.model = model
		self.device = device
		# if self.device >= 0:
		# 	self.model.to_gpu(self.device)
		# self._optimizers['main'].setup(self.model)
		self.no_of_classes=no_of_classes
	def update_core(self):
		image, labels = self.converter(self.get_iterator('main').next())
		image = Variable(image)
		if self.device >= 0:
			image.to_gpu(self.device)
		cl_output = self._optimizers['main'].target.classify(image)
		xp = get_array_module(cl_output.data)

		target = xp.asarray([[0]*self.no_of_classes]*cl_output.shape[0])
		for i in range(labels.shape[0]):
			gt_labels = np.unique(labels[i]).astype(np.int32)[2:] - 1 # Not considering -1 & 0
			target[i][gt_labels] = 1
		loss = F.sigmoid_cross_entropy(cl_output, target, normalize=True)
		report({'Loss':loss}, self.get_optimizer('main').target)
		self._optimizers['main'].target.cleargrads()
		loss.backward()
		self._optimizers['main'].update()



