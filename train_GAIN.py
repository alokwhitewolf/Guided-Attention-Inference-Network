import os
import argparse
import chainer
from models.fcn8 import FCN8s
from chainercv.datasets import VOCSemanticSegmentationDataset
from chainer.iterators import SerialIterator
from chainer.training.trainer import Trainer
from chainer.training import extensions
from chainer.optimizers import Adam
from updater import VOC_GAIN_Updater

def main():
	parser = argparse.ArgumentParser(
	formatter_class=argparse.ArgumentDefaultsHelpFormatter)
	parser.add_argument('--gpu', type=int, default=0, help='gpu id')
	parser.add_argument('--modelfile', help='pretrained model file of FCN8')
	parser.add_argument('--lr', type=float, default=1e-7, help='init learning rate')
	parser.add_argument('--name', type=str, default='exp', help='name of the experiment')
	parser.add_argument('--resume', type=int, default=0, help='resume training or not')
	parser.add_argument('--snapshot', type=str, help='snapshot file to resume from')
	parser.add_argument('--lambda1', default=5, type=float, help='lambda1 param')
	parser.add_argument('--lambda2', default=1, type=float, help='lambda2 param')
	parser.add_argument('--lambda3', default=1.5, type=float, help='lambda3 param')

	args = parser.parse_args()


	resume = args.resume
	device = args.gpu

	if resume:
		load_snapshot_path = args.snapshot
		load_model_path = args.modelfile
	else:
		pretrained_model_path = args.modelfile

	experiment = args.name
	lr = args.lr
	optim = Adam
	training_interval = (20000, 'iteration')
	snapshot_interval = (1000, 'iteration')
	lambd1 = args.lambda1
	lambd2 = args.lambda2
	lambd3 = args.lambda3
	updtr = VOC_GAIN_Updater

	os.makedirs('result/'+experiment, exist_ok=True)
	f = open('result/'+experiment+'/details.txt', "w+")
	f.write("lr - "+str(lr)+"\n")
	f.write("optimizer - "+str(optim)+"\n")
	f.write("lambd1 - "+str(lambd1)+"\n")
	f.write("lambd2 - "+str(lambd2)+"\n")
	f.write("lambd3 - "+str(lambd3)+"\n")
	f.write("training_interval - "+str(training_interval)+"\n")
	f.write("Updater - "+str(updtr)+"\n")
	f.close()

	if resume:
		model = FCN8s()
		chainer.serializers.load_npz(load_model_path, model)
	else:
		model = FCN8s()
		chainer.serializers.load_npz(pretrained_model_path, model)


	if device >= 0:
		model.to_gpu(device)
	dataset = VOCSemanticSegmentationDataset()
	iterator = SerialIterator(dataset, 1, shuffle=False)

	optimizer = Adam(alpha=lr)
	optimizer.setup(model)

	updater = updtr(iterator, optimizer, device=device, lambd1=lambd1, lambd2=lambd2)
	trainer = Trainer(updater, training_interval)
	log_keys = ['epoch', 'iteration', 'main/AM_Loss', 'main/CL_Loss', 'main/TotalLoss']
	trainer.extend(extensions.LogReport(log_keys, (10, 'iteration'), log_name='log'+experiment))
	trainer.extend(extensions.PrintReport(log_keys), trigger=(100, 'iteration'))
	trainer.extend(extensions.ProgressBar(training_length=training_interval, update_interval=100))
	trainer.extend(extensions.snapshot(filename='snapshot'+experiment), trigger=snapshot_interval)
	trainer.extend(extensions.snapshot_object(trainer.updater._optimizers['main'].target, "model"+experiment), trigger=snapshot_interval)
	trainer.extend(extensions.PlotReport(['main/AM_Loss'], 'iteration',(20, 'iteration'), file_name=experiment+'/am_loss.png', grid=True, marker=" "))
	trainer.extend(extensions.PlotReport(['main/CL_Loss'], 'iteration',(20, 'iteration'), file_name=experiment+'/cl_loss.png', grid=True, marker=" "))
	trainer.extend(
		extensions.PlotReport(['main/SG_Loss'], 'iteration', (20, 'iteration'), file_name=experiment + '/sg_loss.png',grid=True, marker=" "))
	trainer.extend(extensions.PlotReport(['main/TotalLoss'], 'iteration',(20, 'iteration'), file_name=experiment+'/total_loss.png', grid=True, marker=" "))
	trainer.extend(extensions.PlotReport(log_keys[2:], 'iteration',(20, 'iteration'), file_name=experiment+'/all_loss.png', grid=True, marker=" "))

	if resume:
		chainer.serializers.load_npz(load_snapshot_path, trainer)
	print("Running - - ", experiment)
	print('initial lr ',lr)
	print('optimizer ', optim)
	print('lambd1 ', lambd1)
	print('lambd2 ', lambd2)
	print('lambd3', lambd3)
	trainer.run()

if __name__ == "__main__":
	main()


