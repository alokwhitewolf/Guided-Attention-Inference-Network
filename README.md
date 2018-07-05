# Guided Attention for FCN

## About
Chainer implementation of Tell Me Where To Look.
This is an experiment to apply Guided Attention Inference Network(GAIN) as presented in <a href='https://arxiv.org/abs/1802.10171'>Tell Me Where To Look</a>to Fully Convolutional Networks(FCN) used for segmentation purposes. The trained FCN8s model is fine tuned using guided attention.

## GAIN
 GAIN is based on supervising the attention maps that is produced when we train the network for
the task of interest.

![Image](media/gain.png)
## FCN
Fully Convolutional Networks is a network architecture that consists of convolution layers followed by deconvolutions to
give the segmentation output

![Image](media/fcn.png)
## Approach

* We take the fully trained FCN8 network and add a average pooling and fully connected layers after its convolutional layers. We freeze the convolutional layers and
train the fully connected networks to classify for the objects. We do this in order to get GradCAMs for a particular class to be later used during GAIN

![Image](media/modification.jpg)

* Next we train the network as per the GAIN update rule. However in this implementation I have also  considered the segmentation loss along with the
GAIN updates/loss. This is because, I found using only the GAIN updates though did lead to convergence of losses, but also resulted in quite a significant dip in segmentation accuracies. In this step, the fully connected ayers are freezed and are not updated.

## Loss Curves
### For classification training
![Image](media/classification_loss.png)

### Segmentation Loss during GAIN updates
![Image](media/sg_loss.png)


## Qualitative Results
| Original Image | PreTrained GCAMs | Post GAIN GCAMs |
|:-------------:|:--------:|:--------------:|
![Image](media/example2.png)

![Image](media/example3.png)

![Image](media/example4.png)

![Image](media/example5.png)


## Quantitative Results


### For FCN8s

| Implementation | Accuracy | Accuracy Class | Mean IU | FWAVACC | Model File |
|:--------------:|:--------:|:--------------:|:-------:|:-------:|:----------:|
| [Original](https://github.com/shelhamer/fcn.berkeleyvision.org/tree/master/voc-fcn8s) | 91.2212 | 77.6146 | 65.5126 | 84.5445 | [`fcn8s_from_caffe.npz`](https://drive.google.com/uc?id=0B9P1L--7Wd2vb0cxV0VhcG1Lb28) |
| Experimental| 90.5962 | **80.4099** | 64.6869 | 83.9952 | [`model.npz`]() |

## How to use
```bash
pip install chainer
pip install chaibercv
pip install cupy
pip install fcn
```
Training
--------
For training the classifier, <a href='https://drive.google.com/uc?id=0B9P1L--7Wd2vWG5MeUEwWmxudU0'>download</a>. the pretrained FCN8s chainer model
```bash
train_classifier.py --modelfile <path to the downloaded pre trained model>
```
For GAIN updates,
```bash
train_GAIN.py --mmodelfile <path to the trained model with trained classifier>
```

The accuracy of original implementation is computed with (`evaluate.py`)
## To Do
[x] Finetune hyperparameters

[x] Push Visualization Code

## Credits
The original FCN module and the fcn package is courtesy of <a href='https://github.com/wkentaro/fcn'>wkentaro</a>
