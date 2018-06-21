import os

from chainer.dataset import download
from chainer.serializers import npz
import chainer.functions as F


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
