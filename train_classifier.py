import argparse
import os
import fcn
import chainer
from models.fcn8 import FCN8s
from chainercv.datasets import VOCSemanticSegmentationDataset
from chainer.iterators import SerialIterator
from chainer.training.trainer import Trainer
from chainer.training import extensions
from chainer.optimizers import Adam, SGD
from updater import VOC_ClassificationUpdater

import matplotlib
matplotlib.use('Agg')

def main():
	parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
	parser.add_argument('--device', type=int, default=-1, help='gpu id')
	parser.add_argument('--lr_init', type=float, default=5*1e-5, help='init learning rate')
	# parser.add_argument('--lr_trigger', type=float, default=5, help='trigger to decreace learning rate')
	# parser.add_argument('--lr_target', type=float, default=5*1e-5, help='target learning rate')
	# parser.add_argument('--lr_factor', type=float, default=.75, help='decay factor')
	parser.add_argument('--name', type=str, default='classifier', help='name of the experiment')
	parser.add_argument('--resume', type=bool, default=False, help='resume training or not')
	parser.add_argument('--snapshot', type=str, help='snapshot file of the trainer to resume from')

	args = parser.parse_args()

	resume = args.resume
	device = args.device

	if resume:
		load_snapshot_path = args.snapshot

	experiment = args.name
	lr_init = args.lr_init
	# lr_target = args.lr_target
	# lr_factor = args.lr_factor
	# lr_trigger_interval = (args.lr_trigger, 'epoch')


	os.makedirs('result/'+experiment, exist_ok=True)
	f = open('result/'+experiment+'/details.txt',"w+")
	f.write("lr - "+str(lr_init)+"\n")
	f.write("optimizer - "+str(Adam))
	# f.write("lr_trigger_interval - "+str(lr_trigger_interval)+"\n")
	f.close()

	if not resume:
		# Add the FC layers to original FCN for GAIN
		model_own = FCN8s()
		model_original = fcn.models.FCN8s()
		model_file = fcn.models.FCN8s.download()
		chainer.serializers.load_npz(model_file, model_original)

		for layers in model_original._children:
			setattr(model_own, layers, getattr(model_original, layers))
		del(model_original, model_file)


	else:
		model_own = FCN8s()

	if device>=0:
		model_own.to_gpu(device)

	dataset = VOCSemanticSegmentationDataset()
	iterator = SerialIterator(dataset, 1)
	optimizer = Adam(alpha=lr_init)
	optimizer.setup(model_own)

	updater = VOC_ClassificationUpdater(iterator, optimizer, device=device)
	trainer = Trainer(updater, (100, 'epoch'))
	log_keys = ['epoch', 'iteration', 'main/Loss']
	trainer.extend(extensions.LogReport(log_keys, (100, 'iteration'), log_name='log_'+experiment))
	trainer.extend(extensions.PrintReport(log_keys), trigger=(100, 'iteration'))
	trainer.extend(extensions.snapshot(filename=experiment+"_snapshot_{.updater.iteration}"), trigger=(5, 'epoch'))
	trainer.extend(extensions.snapshot_object(trainer.updater._optimizers['main'].target, experiment+"_model_{.updater.iteration}"), trigger=(5, 'epoch'))
	trainer.extend(extensions.PlotReport(['main/Loss'], 'iteration',(100, 'iteration'), file_name='trainer_'+experiment+'/loss.png', grid=True, marker=" "))
	
	# trainer.extend(extensions.ExponentialShift('lr', lr_factor, target=lr_target), trigger=lr_trigger_interval)
	if resume:
		chainer.serializers.load_npz(load_snapshot_path, trainer)
	
	print("Running - - ", experiment)
	print('initial lr ', lr_init)
	# print('lr_trigger_interval ', lr_trigger_interval)
	trainer.run()

if __name__ =="__main__":
	main()

