import argparse
import chainer
from chainer import cuda
import fcn
import numpy as np
import tqdm
from models.fcn8 import FCN8s


def evaluate():
	parser = argparse.ArgumentParser(
	formatter_class=argparse.ArgumentDefaultsHelpFormatter)
	parser.add_argument('--file', type=str, help='model file path')

	args = parser.parse_args()
	file = args.file
	print("evaluating: ",file)
	dataset = fcn.datasets.VOC2011ClassSeg('seg11valid')
	n_class = len(dataset.class_names)

	model = FCN8s()
	chainer.serializers.load_npz(file, model)

	gpu = 0

	if gpu >= 0:
		cuda.get_device(gpu).use()
		model.to_gpu()

	lbl_preds, lbl_trues = [], []
	for i in tqdm.trange(len(dataset)):
		datum, lbl_true = fcn.datasets.transform_lsvrc2012_vgg16(
			dataset.get_example(i))
		x_data = np.expand_dims(datum, axis=0)
		if gpu >= 0:
			x_data = cuda.to_gpu(x_data)

		with chainer.no_backprop_mode():
			x = chainer.Variable(x_data)
			with chainer.using_config('train', False):
				model(x)
				lbl_pred = chainer.functions.argmax(model.score, axis=1)[0]
				lbl_pred = chainer.cuda.to_cpu(lbl_pred.data)

		lbl_preds.append(lbl_pred)
		lbl_trues.append(lbl_true)

	acc, acc_cls, mean_iu, fwavacc = fcn.utils.label_accuracy_score(lbl_trues, lbl_preds, n_class)
	print('Accuracy: %.4f' % (100 * acc))
	print('AccClass: %.4f' % (100 * acc_cls))
	print('Mean IoU: %.4f' % (100 * mean_iu))
	print('Fwav Acc: %.4f' % (100 * fwavacc))
if __name__ == '__main__':
	evaluate()