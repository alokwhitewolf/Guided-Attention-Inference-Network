import collections
import os
import sys

import chainer
import chainer.functions as F
import chainer.links as L
from chainer.dataset import download
from chainer.serializers import npz


class ResNet152_GAIN(chainer.Chain):

    def __init__(self):
        super(ResNet152_GAIN, self).__init__()

        with self.init_scope():
            self.conv1 = L.Convolution2D(3, 64, 7, 2, 3)
            self.bn1 = L.BatchNormalization(64)
            self.res2 = BuildingBlock(3, 64, 64, 256, 1)
            self.res3 = BuildingBlock(8, 256, 128, 512, 2)
            self.res4 = BuildingBlock(36, 512, 256, 1024, 2)
            self.res5 = BuildingBlock(3, 1024, 512, 2048, 2)
            self.fc6 = L.Linear(2048, 1000)

        _retrieve('ResNet-152-model.npz', 'ResNet-152-model.caffemodel', self)

        self.size = 224
        self.functions = collections.OrderedDict([
            ('conv1', [self.conv1, self.bn1, F.relu]),
            ('pool1', [lambda x: F.max_pooling_2d(x, 3, 2)]),
            ('res2', [self.res2]),
            ('res3', [self.res3]),
            ('res4', [self.res4]),
            ('res5', [self.res5]),
            ('pool5', [lambda x: F.average_pooling_2d(x, x.shape[2:], 1)]),
            ('fc6', [self.fc6]),
            ('prob', [F.softmax]),
        ])

    def convert_caffemodel_to_npz(self, path_caffemodel, path_npz):
        # As CaffeFunction uses shortcut symbols,
        # we import CaffeFunction here.
        from chainer.links.caffe.caffe_function import CaffeFunction
        caffemodel = CaffeFunction(path_caffemodel)
        _transfer_resnet152(caffemodel, self)
        npz.save_npz(path_npz, self, compression=False)

class BuildingBlock(chainer.Chain):

    def __init__(self, n_layer, in_channels, mid_channels, out_channels,
                 stride):
        super(BuildingBlock, self).__init__()
        with self.init_scope():
            self.a = BottleneckA(
                in_channels, mid_channels, out_channels, stride)
            self._forward = ["a"]
            for i in range(n_layer - 1):
                name = 'b{}'.format(i + 1)
                bottleneck = BottleneckB(out_channels, mid_channels)
                setattr(self, name, bottleneck)
                self._forward.append(name)

    def __call__(self, x):
        for name in self._forward:
            l = getattr(self, name)
            x = l(x)
        return x

    @property
    def forward(self):
        return [getattr(self, name) for name in self._forward]


class BottleneckA(chainer.Chain):

    def __init__(self, in_channels, mid_channels, out_channels, stride=2):
        super(BottleneckA, self).__init__()
        with self.init_scope():
            self.conv1 = L.Convolution2D(
                in_channels, mid_channels, 1, 1, 0, nobias=True)
            self.bn1 = L.BatchNormalization(mid_channels)
            self.conv2 = L.Convolution2D(
                mid_channels, mid_channels, 3, stride, 1, nobias=True)
            self.bn2 = L.BatchNormalization(mid_channels)
            self.conv3 = L.Convolution2D(
                mid_channels, out_channels, 1, 1, 0, nobias=True)
            self.bn3 = L.BatchNormalization(out_channels)
            self.conv4 = L.Convolution2D(
                in_channels, out_channels, 1, stride, 0, nobias=True)
            self.bn4 = L.BatchNormalization(out_channels)

        self.functions = collections.OrderedDict([
            ('conv1', [self.conv1, self.bn1, F.relu]),
            ('conv2', [self.conv2, self.bn2, F.relu]),
            ('conv3', [self.conv3, self.bn3]),
            ('proj4', [self.conv4, self.bn4]),
            ('relu5', [F.add, F.relu]),
        ])

    def __call__(self, x):
        h = x
        for key, funcs in self.functions.items():
            if key == 'proj4':
                for func in funcs:
                    x = func(x)
            else:
                for func in funcs:
                    if func is F.add:
                        h = func(h, x)
                    else:
                        h = func(h)
        return h


class BottleneckB(chainer.Chain):

    def __init__(self, in_channels, mid_channels):
        super(BottleneckB, self).__init__()
        with self.init_scope():
            self.conv1 = L.Convolution2D(
                in_channels, mid_channels, 1, 1, 0, nobias=True)
            self.bn1 = L.BatchNormalization(mid_channels)
            self.conv2 = L.Convolution2D(
                mid_channels, mid_channels, 3, 1, 1, nobias=True)
            self.bn2 = L.BatchNormalization(mid_channels)
            self.conv3 = L.Convolution2D(
                mid_channels, in_channels, 1, 1, 0, nobias=True)
            self.bn3 = L.BatchNormalization(in_channels)

        self.functions = collections.OrderedDict([
            ('conv1', [self.conv1, self.bn1, F.relu]),
            ('conv2', [self.conv2, self.bn2, F.relu]),
            ('conv3', [self.conv3, self.bn3]),
            ('relu4', [F.add, F.relu]),
        ])

    def __call__(self, x):
        h = x
        for key, funcs in self.functions.items():
            for func in funcs:
                if func is F.add:
                    h = func(h, x)
                else:
                    h = func(h)
        return h


def _transfer_components(src, dst_conv, dst_bn, bname, cname):
    src_conv = getattr(src, 'res{}_branch{}'.format(bname, cname))
    src_bn = getattr(src, 'bn{}_branch{}'.format(bname, cname))
    src_scale = getattr(src, 'scale{}_branch{}'.format(bname, cname))
    dst_conv.W.data[:] = src_conv.W.data
    dst_bn.avg_mean[:] = src_bn.avg_mean
    dst_bn.avg_var[:] = src_bn.avg_var
    dst_bn.gamma.data[:] = src_scale.W.data
    dst_bn.beta.data[:] = src_scale.bias.b.data


def _transfer_bottleneckA(src, dst, name):
    _transfer_components(src, dst.conv1, dst.bn1, name, '2a')
    _transfer_components(src, dst.conv2, dst.bn2, name, '2b')
    _transfer_components(src, dst.conv3, dst.bn3, name, '2c')
    _transfer_components(src, dst.conv4, dst.bn4, name, '1')


def _transfer_bottleneckB(src, dst, name):
    _transfer_components(src, dst.conv1, dst.bn1, name, '2a')
    _transfer_components(src, dst.conv2, dst.bn2, name, '2b')
    _transfer_components(src, dst.conv3, dst.bn3, name, '2c')


def _transfer_block(src, dst, names):
    _transfer_bottleneckA(src, dst.a, names[0])
    for i, name in enumerate(names[1:]):
        dst_bottleneckB = getattr(dst, 'b{}'.format(i + 1))
        _transfer_bottleneckB(src, dst_bottleneckB, name)


def _transfer_resnet152(src, dst):
    dst.conv1.W.data[:] = src.conv1.W.data
    dst.bn1.avg_mean[:] = src.bn_conv1.avg_mean
    dst.bn1.avg_var[:] = src.bn_conv1.avg_var
    dst.bn1.gamma.data[:] = src.scale_conv1.W.data
    dst.bn1.beta.data[:] = src.scale_conv1.bias.b.data

    _transfer_block(src, dst.res2, ['2a', '2b', '2c'])
    _transfer_block(src, dst.res3,
                    ['3a'] + ['3b{}'.format(i) for i in range(1, 8)])
    _transfer_block(src, dst.res4,
                    ['4a'] + ['4b{}'.format(i) for i in range(1, 36)])
    _transfer_block(src, dst.res5, ['5a', '5b', '5c'])

    dst.fc6.W.data[:] = src.fc1000.W.data
    dst.fc6.b.data[:] = src.fc1000.b.data


def _make_npz(path_npz, path_caffemodel, model):
    sys.stderr.write(
        'Now loading caffemodel (usually it may take few minutes)\n')
    sys.stderr.flush()
    if not os.path.exists(path_caffemodel):
        raise IOError(
            'The pre-trained caffemodel does not exist. Please download it '
            'from \'https://github.com/KaimingHe/deep-residual-networks\', '
            'and place it on {}'.format(path_caffemodel))
    model.convert_caffemodel_to_npz(path_caffemodel, path_npz)
    npz.load_npz(path_npz, model)
    return model


def _retrieve(name_npz, name_caffemodel, model):
    root = download.get_dataset_directory('pfnet/chainer/models/')
    path = os.path.join(root, name_npz)
    path_caffemodel = os.path.join(root, name_caffemodel)
    return download.cache_or_load_file(
        path, lambda path: _make_npz(path, path_caffemodel, model), lambda path: npz.load_npz(path, model))
