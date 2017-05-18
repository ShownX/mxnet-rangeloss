from __future__ import print_function
import logging
import mxnet as mx
import argparse
import numpy
import matplotlib.pyplot as plt
import seaborn as sns
import os

logging.basicConfig(format="[%(asctime)s][%(levelname)s] %(message)s")
logger = logging.getLogger()
logger.setLevel(logging.INFO)


def inference():
    data = mx.sym.Variable('data')
    label = mx.sym.Variable('softmax_label')
    conv1 = mx.sym.Convolution(data=data, kernel=(5, 5), num_filter=32)
    relu1 = mx.sym.Activation(data=conv1, act_type='relu')
    pool1 = mx.sym.Pooling(data=relu1, kernel=(2, 2), stride=(2, 2), pool_type='max')
    conv2 = mx.sym.Convolution(data=pool1, kernel=(5, 5), num_filter=64)
    relu2 = mx.sym.Activation(data=conv2, act_type='relu')
    pool2 = mx.sym.Pooling(data=relu2, kernel=(2, 2), stride=(2, 2), pool_type='max')
    fc3 = mx.sym.FullyConnected(data=pool2, num_hidden=256)
    relu3 = mx.sym.Activation(data=fc3, act_type='relu')
    embedding = mx.sym.FullyConnected(data=relu3, num_hidden=2, name='embedding')
    if not args.useSoftmaxOnly:
        range_loss = mx.sym.Custom(data=embedding, label=label, num_hidden=10, op_type='RangeLoss')
    else:
        range_loss = 0

    softmax_loss = mx.sym.SoftmaxOutput(data=embedding, label=label)

    return softmax_loss + args.l*range_loss


def train():
    ctx = mx.gpu(args.gpu) if args.gpu >= 0 else mx.cpu()
    train_iter = mx.io.MNISTIter(
        image='data/train-images-idx3-ubyte',
        label='data/train-labels-idx1-ubyte',
        input_shape=(1, 28, 28),
        mean_r=128,
        scale=1. / 128,
        batch_size=args.batch_size,
        shuffle=True)
    val_iter = mx.io.MNISTIter(
        image='data/t10k-images-idx3-ubyte',
        label='data/t10k-labels-idx1-ubyte',
        input_shape=(1, 28, 28),
        mean_r=128,
        scale=1. / 128,
        batch_size=args.batch_size)

    symbol = inference()

    mod = mx.mod.Module(symbol=symbol, context=ctx)

    num_examples = 60000
    epoch_size = int(num_examples/args.batch_size)
    optim_params = {
        'learning_rate': args.lr,
        'momentum': 0.9,
        'wd': 0.0005,
        'lr_scheduler': mx.lr_scheduler.FactorScheduler(step=10*epoch_size, factor=0.1)
    }

    mod.fit(train_data=train_iter, eval_data=val_iter, eval_metric=mx.metric.Accuracy(), initializer=mx.init.Xavier(),
            optimizer='sgd', optimizer_params=optim_params, num_epoch=args.max_epoch,
            batch_end_callback=mx.callback.Speedometer(args.batch_size, 50),
            epoch_end_callback=mx.callback.do_checkpoint('model/mnist'))


def test():
    ctx = mx.gpu(args.gpu) if args.gpu >= 0 else mx.cpu()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_epoch', type=int, default=20, help='number of epoch')
    parser.add_argument('--batch_size', type=int, default=128, help='batch size')
    parser.add_argument('--gpu', type=int, default=0, help='gpu index')
    parser.add_argument('--train', action='store_true', help='train mnist')
    parser.add_argument('--test', action='store_true', help='train mnist and plot')
    parser.add_argument('--useSoftmaxOnly', type=bool, default=False, help='use softmax loss')
    parser.add_argument('--lr', type=float, default=0.1, help='learning rate')
    parser.add_argument('--l', type=float, default=5, help='lambda to balance the loss')
    parser.add_argument('--m', type=float, default=20000, help='margin parameter')
    parser.add_argument('--a', type=float, default=1, help='alpha parameter')
    parser.add_argument('--b', type=float, default=1, help='beta parameter')
    parser.add_argument('--k', type=float, default=2, help='k parameter')
    args = parser.parse_args()

    if args.train:
        train()

    if args.test:
        test()