import argparse
import os
import fcn
import chainer
from models.fcn8 import FCN8s
from chainercv.datasets import VOCSemanticSegmentationDataset
from chainer.iterators import SerialIterator
from chainer.training.trainer import Trainer
from chainer.training import extensions
from chainer.optimizers import Adam
from updater import VOC_ClassificationUpdater

def main():
	parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
	parser.add_argument('--gpu', type=int, default=0, help='gpu id')
	parser.add_argument('--modelfile', help='pretrained model file of FCN8')
	parser.add_argument('--lr', type=float, default=5*1e-4, help='init learning rate')
	parser.add_argument('--name', type=str, default='exp', help='init learning rate')
	parser.add_argument('--resume', type=int, default=0, help='resume training or not')
	parser.add_argument('--snapshot', type=str, help='snapshot file to resume from')

	args = parser.parse_args()

	resume = args.resume
	device = args.gpu

	if resume:
		load_snapshot_path = args.snapshot
		load_model_path = args.modelfile
	else:
		load_model_path = args.modelfile

	experiment = args.name
	lr = args.lr
	lr_trigger_interval = (5, 'epoch')
	optim = Adam

	os.makedirs('result/'+experiment, exist_ok=True)
	f = open('result/'+experiment+'/details.txt',"w+")
	f.write("lr - "+str(lr)+"\n")
	f.write("optimizer - "+str(optim))
	f.write("lr_trigger_interval - "+str(lr_trigger_interval)+"\n")
	f.close()

	if not resume:
		# Add the FC layers to original FConvN for GAIN
		model_own = FCN8s()
		model_original = fcn.models.FCN8s()
		model_file = fcn.models.FCN8s.download()
		chainer.serializers.load_npz(model_file, model_original)

		for layers in model_original._children:
			setattr(model_own, layers, getattr(model_original, layers))
		del(model_original, model_file)


	else:
		model_own = FCN8s()
		chainer.serializers.load_npz(load_model_path, model_own)

	if device>=0:
		model_own.to_gpu(device)

	dataset = VOCSemanticSegmentationDataset()
	iterator = SerialIterator(dataset, 1)
	optimizer = Adam(alpha=lr)
	optimizer.setup(model_own)

	updater = VOC_ClassificationUpdater(iterator, optimizer, device=device)
	trainer = Trainer(updater, (50, 'epoch'))
	log_keys = ['epoch', 'iteration', 'main/Loss']
	trainer.extend(extensions.LogReport(log_keys, (100, 'iteration'), log_name='log'+experiment))
	trainer.extend(extensions.PrintReport(log_keys), trigger=(100, 'iteration'))
	trainer.extend(extensions.ProgressBar(training_length=(50, 'epoch'), update_interval=500))
	trainer.extend(extensions.snapshot(filename='snapshot'+experiment), trigger=(5, 'epoch'))
	trainer.extend(extensions.snapshot_object(trainer.updater._optimizers['main'].target, "model"+experiment), trigger=(5, 'epoch'))
	trainer.extend(extensions.PlotReport(['main/Loss'], 'iteration',(100, 'iteration'), file_name=experiment+'/loss.png', grid=True, marker=" "))

	if resume:
		chainer.serializers.load_npz(load_snapshot_path, trainer)
	print("Running - - ", experiment)
	print('initial lr ', lr)
	print('lr_trigger_interval ', lr_trigger_interval)
	trainer.run()

if __name__ =="__main__":
	main()

