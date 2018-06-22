import chainer
import chainer.functions as F


class GAIN(chainer.Chain):
	def __init__(self):
		super(GAIN, self).__init__()


	def stream_cl(self, inp, cl_target, final_conv_layer, grad_target_layer='prob', class_id=None):
		h = chainer.Variable(inp)
		# activations = {'input': h}

		for key, funcs in self.functions.items():
			for func in funcs:
				h = func(h)
			if key == grad_target_layer:
				break

		gcam = self.get_gcam(h, getattr(self, final_conv_layer), class_id)
		loss = F.softmax_cross_entropy(h, cl_target)
		return loss, gcam


	def stream_am(self, masked_image, cl_target, class_id):
		pass


	def get_gcam(self, end_output, conv_link, class_id=None):
		self.cleargrads()
		self.set_init_grad(end_output, class_id)
		end_output.backward()

		grad = conv_link.W.grad_var
		grad = F.average_pooling_2d(grad, grad.shape[2], 1)
		grad = F.expand_dims(F.reshape(grad, (grad.shape[0]*grad.shape[1], grad.shape[2], grad.shape[3])), 0)

		weights = conv_link.W
		weights = F.expand_dims(F.reshape(weights, (weights.shape[0]*weights.shape[1], weights.shape[2], weights.shape[3])), 0)

		self.cleargrads()
		return F.resize_images(F.convolution_2d(weights, grad, None, 1, 0), (self.size, self.size))


	def set_init_grad(self, var, class_id=None):
		var.grad = self.xp.zeros_like(var.data)
		if class_id is None:
			var.grad[0][F.argmax(var).data] = 1
		else:
			var.grad[0][class_id] = 1

	def mask_image(self, gcam, img, sigma, w):
		mask = F.squeeze(F.sigmoid(-w * (gcam - sigma)))
		for channel_depth in range(img.shape[-3]):
			img[0][channel_depth]*mask
		return img


