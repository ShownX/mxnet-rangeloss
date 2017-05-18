from __future__ import print_function
import logging
import mxnet as mx
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects
import seaborn as sns
import os

logging.basicConfig(level=logging.INFO)


def inference():
    data = mx.sym.Variable('data')
    label = mx.sym.Variable('softmax_label')
    conv1 = mx.sym.Convolution(data=data, kernel=(5, 5), num_filter=32)
    relu1 = mx.sym.Activation(data=conv1, act_type='relu')
    pool1 = mx.sym.Pooling(data=relu1, kernel=(2, 2), stride=(2, 2), pool_type='max')
    conv2 = mx.sym.Convolution(data=pool1, kernel=(5, 5), num_filter=64)
    relu2 = mx.sym.Activation(data=conv2, act_type='relu')
    pool2 = mx.sym.Pooling(data=relu2, kernel=(2, 2), stride=(2, 2), pool_type='max')
    flatten = mx.sym.Flatten(data=pool2, name='flatten')
    fc3 = mx.sym.FullyConnected(data=flatten, num_hidden=256)
    relu3 = mx.sym.Activation(data=fc3, act_type='relu')
    embedding = mx.sym.FullyConnected(data=relu3, num_hidden=2, name='embedding')

    fc4 = mx.sym.FullyConnected(data=embedding, num_hidden=10, no_bias=True)

    if not args.useSoftmaxOnly:
        softmax_loss = mx.sym.SoftmaxOutput(data=fc4, label=label, name='softmax')
        range_loss = mx.sym.Custom(data=fc4, label=label, num_hidden=10, op_type='RangeLoss')

        return softmax_loss + args.l * range_loss
    else:
        softmax_loss = mx.sym.SoftmaxOutput(data=fc4, label=label, name='softmax')
        return softmax_loss


def plot_mnist(feature, label, fname):
    assert feature.shape[1] == 2
    names = dict()
    for i in range(10):
        names[i] = str(i)
    palette = np.array(sns.color_palette("hls", 10))
    f = plt.figure(figsize=(8, 8))
    ax = plt.subplot(aspect='equal')
    ax.scatter(feature[:, 0], feature[:, 1], lw=0, s=40, c=palette[label.astype(np.int)])
    ax.axis('off')
    ax.axis('tight')

    # We add the labels for each digit.
    txts = []
    for i in range(10):
        # Position of each label.
        xtext, ytext = np.median(feature[label == i, :], axis=0)
        txt = ax.text(xtext, ytext, names[i])
        txt.set_path_effects([
            PathEffects.Stroke(linewidth=5, foreground="w"),
            PathEffects.Normal()])
        txts.append(txt)

    plt.show()
    f.savefig(fname)


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

    acc = mx.metric.Accuracy()

    mod = mx.mod.Module(symbol=symbol, context=ctx)

    num_examples = 60000
    epoch_size = int(num_examples/args.batch_size)
    optim_params = {
        'learning_rate': args.lr,
        'momentum': 0.9,
        'wd': 0.0005,
        'lr_scheduler': mx.lr_scheduler.FactorScheduler(step=10*epoch_size, factor=0.1)
    }

    if not os.path.exists('model'):
        os.mkdir('model')

    mod.fit(train_data=train_iter, eval_data=val_iter, eval_metric=acc, initializer=mx.init.Xavier(),
            optimizer='sgd', optimizer_params=optim_params, num_epoch=args.max_epoch,
            batch_end_callback=mx.callback.Speedometer(args.batch_size, 50),
            epoch_end_callback=mx.callback.do_checkpoint(args.model_prefix))


def test():
    ctx = mx.gpu(args.gpu) if args.gpu >= 0 else mx.cpu()
    val_iter = mx.io.MNISTIter(
            image='data/t10k-images-idx3-ubyte',
            label='data/t10k-labels-idx1-ubyte',
            input_shape=(1, 28, 28),
            mean_r=128,
            scale=1. / 128,
            batch_size=1)
    symbol, arg_params, aux_params = mx.model.load_checkpoint(args.model_prefix, args.max_epoch)
    embedding = symbol.get_internals()['embedding_output']
    mod = mx.mod.Module(symbol=embedding, context=ctx, data_names=('data', ))
    mod.bind(data_shapes=[('data', (1, 1, 28, 28))], for_training=False)
    mod.init_params(arg_params=arg_params, aux_params=aux_params)

    embeds = []
    labels = []
    for preds, i_batch, batch in mod.iter_predict(val_iter):
        embeds.append(preds[0].asnumpy())
        labels.append(batch.label[0].asnumpy())

    embeds = np.vstack(embeds)
    labels = np.hstack(labels)

    plot_mnist(embeds, labels, args.plot)


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
    parser.add_argument('--model_prefix', type=str, default='model/mnist', help='model prefix')
    parser.add_argument('--plot', type=str, default='plot/plot-softmax.png', help='plot path')
    args = parser.parse_args()

    if not os.path.exists('model'):
        os.mkdir('model')

    if not os.path.exists('plot'):
        os.mkdir('plot')

    if args.train:
        train()

    if args.test:
        test()
