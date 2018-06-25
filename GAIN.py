import chainer
import chainer.functions as F


class GAIN(chainer.Chain):
	def __init__(self):
		super(GAIN, self).__init__()
		# To override in child class
		self.size = None  # Size of images
		self.GAIN_functions = None  # Refer files in /models
		self.final_conv_layer = None
		self.grad_target_layer = None

	def stream_cl(self, inp, class_id=None):
		h = inp
		for key, funcs in self.GAIN_functions.items():
			for func in funcs:
				h = func(h)
			if key == self.final_conv_layer:
				activation = h
			if key == self.grad_target_layer:
				break

		gcam = self.get_gcam(h, activation, class_id)
		return gcam, h

	def stream_am(self, masked_image):
		h = masked_image
		for key, funcs in self.GAIN_functions.items():
			for func in funcs:
				h = func(h)

		return h

	def stream_ext(self, inp,  class_id=None):
		h = chainer.Variable(inp)
		for key, funcs in self.GAIN_functions.items():
			for func in funcs:
				h = func(h)
			if key == self.final_conv_layer:
				activation = h
			if key == self.grad_target_layer:
				break
			gcam = self.get_gcam(h, activation, class_id)
			mask = self.get_mask(gcam)
			return mask

	def get_gcam(self, end_output, activations, class_id=None):
		self.cleargrads()
		self.set_init_grad(end_output, class_id)
		end_output.backward(retain_grad=True)

		grad = activations.grad_var
		grad = F.average_pooling_2d(grad, grad.shape[2], 1)
		grad = F.expand_dims(F.reshape(grad, (grad.shape[0]*grad.shape[1], grad.shape[2], grad.shape[3])), 0)

		weights = activations
		weights = F.expand_dims(F.reshape(weights, (weights.shape[0]*weights.shape[1], weights.shape[2], weights.shape[3])), 0)


		return F.resize_images(F.relu(F.convolution_2d(weights, grad, None, 1, 0)), (self.size, self.size))

	def set_init_grad(self, var, class_id=None):
		var.grad = self.xp.zeros_like(var.data)
		if class_id is None:
			var.grad[0][F.argmax(var).data] = 1
		else:
			var.grad[0][class_id] = 1

	def mask_image(self, img, mask):
		"""

		:param img: Var of shape [H, W, C]
		:param mask:
		:return:
		"""
		img = F.expand_dims(img.transpose((2, 0, 1)), 0)
		img = F.resize_images(img, (self.size, self.size))
		broadcasted_mask = F.broadcast_to(mask, img.shape)
		to_subtract = img*broadcasted_mask
		return img - to_subtract

	@staticmethod
	def get_mask(gcam, sigma=.5, w=10):
		gcam = (gcam - F.min(gcam).data)/(F.max(gcam) - F.min(gcam)).data
		mask = F.squeeze(F.sigmoid(w * (gcam - sigma)))
		return mask
