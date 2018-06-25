import os
import chainer.functions as F
from chainer.dataset import download
from chainer.serializers import npz
from chainer.backends.cuda import get_array_module
import numpy as np
from PIL import Image
from cupy import get_array_module



def convert_caffemodel_to_npz(path_caffemodel, path_npz):
	from chainer.links.caffe.caffe_function import CaffeFunction
	caffemodel = CaffeFunction(path_caffemodel)
	npz.save_npz(path_npz, caffemodel, compression=False)


def _make_npz(path_npz, url, model):
	path_caffemodel = download.cached_download(url)
	print('Now loading caffemodel (usually it may take few minutes)')
	convert_caffemodel_to_npz(path_caffemodel, path_npz)
	npz.load_npz(path_npz, model)
	return model


def _retrieve(name, url, model):
	root = download.get_dataset_directory('pfnet/chainer/models/')
	path = os.path.join(root, name)
	return download.cache_or_load_file(
		path, lambda path: _make_npz(path, url, model),
		lambda path: npz.load_npz(path, model))


def read_image(path, dtype=np.float32, color=True):
	"""Read an image from a file.
	This function reads an image from given file. The image is CHW format and
	the range of its value is :math:`[0, 255]`. If :obj:`color = True`, the
	order of the channels is RGB.
	Args:
		path (string): A path of image file.
		dtype: The type of array. The default value is :obj:`~numpy.float32`.
		color (bool): This option determines the number of channels.
			If :obj:`True`, the number of channels is three. In this case,
			the order of the channels is RGB. This is the default behaviour.
			If :obj:`False`, this function returns a grayscale image.
	Returns:
		~numpy.ndarray: An image.
	"""

	f = Image.open(path)
	try:
		if color:
			img = f.convert('RGB')
		else:
			img = f.convert('P')
		img = np.asarray(img, dtype=dtype)
	finally:
		if hasattr(f, 'close'):
			f.close()

	return img

def VGGprepare(image=None, path=None, size=(224, 224)):
	"""Converts the given image to the numpy array for VGG models.
	Note that you have to call this method before ``__call__``
	because the pre-trained vgg model requires to resize the given image,
	covert the RGB to the BGR, subtract the mean,
	and permute the dimensions before calling.
	Args:
		image (PIL.Image or numpy.ndarray): Input image.
			If an input is ``numpy.ndarray``, its shape must be
			``(height, width)``, ``(height, width, channels)``,
			or ``(channels, height, width)``, and
			the order of the channels must be RGB.
		size (pair of ints): Size of converted images.
			If ``None``, the given image is not resized.
	Returns:
		numpy.ndarray: The converted output array.
	"""
	if path is not None:
		image = read_image(path)
	if image.ndim == 4:
		image = np.squeeze(image, 0)
	if isinstance(image, np.ndarray):
		if image.ndim == 3:
			if image.shape[0] == 1:
				image = image[0, :, :]
			elif image.shape[0] == 3:
				image = image.transpose((1, 2, 0))
		image = Image.fromarray(image.astype(np.uint8))

	image = image.convert('RGB')
	if size:
		image = image.resize(size)
	image = np.asarray(image, dtype=np.float32)
	image = image[:, :, ::-1]
	image -= np.array(
		[103.939, 116.779, 123.68], dtype=np.float32)
	image = image.transpose((2, 0, 1))
	return np.expand_dims(image, 0)

def VGGprepare_am_input(var):
	xp = get_array_module(var)

	# var = F.resize_images(var, size)
	var = F.transpose(var, (0, 2, 3, 1)) # [[W, H, C]]
	var = F.flip(var, 3)
	var -= xp.array([[103.939, 116.779, 123.68]], dtype=xp.float32)
	var = F.transpose(var, (0, 3, 1, 2))
	return var

