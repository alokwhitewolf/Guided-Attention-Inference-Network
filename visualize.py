import os
import argparse
import chainer
from chainer import Variable
import chainer.functions as F
from models.fcn8 import FCN8s
from chainercv.datasets import VOCSemanticSegmentationDataset
from chainer.iterators import SerialIterator
from chainer.serializers import load_npz
from chainer.backends.cuda import  get_array_module
from chainercv.datasets.voc.voc_utils import voc_semantic_segmentation_label_names

from matplotlib import pyplot as plt
import numpy as np
import cupy as cp


def main()
	parser = argparse.ArgumentParser(
	formatter_class=argparse.ArgumentDefaultsHelpFormatter)
	parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
	parser.add_argument('--pretrained', type=str, help='path to model that has trained classifier but has not been trained through GAIN routine')
	parser.add_argument('--trained', type=str, help='path to model trained through GAIN')
	parser.add_argument('--device', type=int, default=-1, help='gpu id')
	parser.add_argument('--shuffle', type=bool, default=False, help='whether to shuffle dataset')
	parser.add_argument('--whole', type=bool, default=False, help='whether to test for the whole validation dataset')
	parser.add_argument('--no', type=int, default=10, help='if not whole, then no of images to visualize')
	parser.add_argument('--name', type=str, help='name of the subfolder or experiment under which to save')

	args = parser.parse_args()

	pretrained_file = args.pretrained
	trained_file = args.trained
	device = args.device
	shuffle = args.shuffle
	whole = args.whole
	name = args.name
	N = args.no

	dataset = VOCSemanticSegmentationDataset()
	iterator = SerialIterator(dataset, 1, shuffle=shuffle, repeat=False)
	converter = chainer.dataset.concat_examples
	os.makedirs('viz/'+name, exist_ok=True)
	no_of_classes = 20
	device = 0
	pretrained = FCN8s()
	trainer = FCN8s()
	load_npz(pretrained_file, pretrained)
	load_npz(trained_file, trained)
	
	if device >=0:
		pretrained.to_gpu()
		trained.to_gpu()
	i = 0
	
	while not iterator.is_new_epoch:
		
		if not whole and i >= N:
			break

		image, labels = converter(iterator.next())
		image = Variable(image)
		if device >=0:
			image.to_gpu()

		xp = get_array_module(image.data)
		to_substract = np.array((-1, 0))
		noise_classes = np.unique(labels[0]).astype(np.int32)
		target = xp.asarray([[0]*(no_of_classes)])
		gt_labels = np.setdiff1d(noise_classes, to_substract) - 1

		gcam1, cl_scores1, class_id1 = pretrained.stream_cl(image, gt_labels)
		gcam2, cl_scores2, class_id2 = trained.stream_cl(image, gt_labels)

		if device>-0:
			class_id = cp.asnumpy(class_id)
		fig1 = plt.figure(figsize=(20,10))
		ax1= plt.subplot2grid((3, 9), (0, 0), colspan=3, rowspan=3)
		ax1.axis('off')
		ax1.imshow(cp.asnumpy(F.transpose(F.squeeze(image, 0), (1, 2, 0)).data) / 255.)

		ax2= plt.subplot2grid((3, 9), (0, 3), colspan=3, rowspan=3)
		ax2.axis('off')
		ax2.imshow(cp.asnumpy(F.transpose(F.squeeze(image, 0), (1, 2, 0)).data) / 255.)
		ax2.imshow(cp.asnumpy(F.squeeze(gcam1[0], 0).data), cmap='jet', alpha=.5)
		ax2.set_title("For class - "+str(voc_semantic_segmentation_label_names[cp.asnumpy(class_id1[0])+1]), color='teal')

		ax3= plt.subplot2grid((3, 9), (0, 6), colspan=3, rowspan=3)
		ax3.axis('off')
		ax3.imshow(cp.asnumpy(F.transpose(F.squeeze(image, 0), (1, 2, 0)).data) / 255.)
		ax3.imshow(cp.asnumpy(F.squeeze(gcam2[0], 0).data), cmap='jet', alpha=.5)
		ax3.set_title("For class - "+str(voc_semantic_segmentation_label_names[cp.asnumpy(class_id2[0])+1]), color='teal')
		fig1.savefig('viz/'+name+'/'+str(i)+'.png')
		plt.close()
		print(i)
		i += 1

if __name__ =="__main__":
	main()