import os
import mxnet as mx
import numpy as np


class RangeLossOp(mx.operator.CustomOp):
    def __init__(self, margin, k, alpha, beta):
        self.margin = margin
        self.k = k
        self.alpha = alpha
        self.beta = beta

    def compute_top_k(self, features):
        num = features.shape[0]
        dists = np.zeros((num, num))
        for id_1 in range(0, num):
            for id_2 in range(id_1+1, num):
                dist = np.linalg.norm(features[id_1] - features[id_2])
                dists[id_1, id_2] = dist

        dist_array = dists.reshape((1, -1))

        return dist_array.sort()[-self.k:]

    def compute_min_dist(self, centers):
        num = centers.shape[0]
        dists = np.ones((num, num)) * -1
        for id_1 in range(0, num):
            for id_2 in range(id_1+1, num):
                dist = np.linalg.norm(centers[id_1] - centers[id_2])
                dists[id_1, id_2] = dist

        dist_array = dists.reshape((1, -1))
        dist_array = dist_array[np.where(dist_array > 0)]

        return dist_array.sort()[0]

    def forward(self, is_train, req, in_data, out_data, aux):
        features = in_data[0].asnumpy()
        labels = in_data[1].asnumpy()

        unique_labels, counts = np.unique(labels, return_counts=True)

        centers = np.zeros((unique_labels.shape[0], features.shape[1]))
        d = np.zeros((unique_labels.shape[0], self.k))

        l_r = np.zeros((unique_labels.shape[0]))

        for idx, l in enumerate(unique_labels):
            indices = np.where(labels == l)
            features_l = features[indices, :]
            centers[idx, :] = np.mean(features_l, axis=0)
            d[idx, :] = self.compute_top_k(features_l)
            l_r[idx] = self.k / np.sum(d[idx, :])

        l_intra = np.sum(l_r)
        d_center = self.compute_min_dist(centers)
        l_inter = max(self.margin - d_center, 0)
        loss = l_intra * self.alpha + l_inter * self.beta
        self.assign(out_data[0], req[0], mx.nd.array(loss))

    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        features = in_data[0].asnumpy()
        labels = in_data[1].asnumpy().ravel().astype(np.int)

        unique_labels, counts = np.unique(labels, return_counts=True)

        centers = np.zeros((unique_labels.shape[0], features.shape[1]))
        d = np.zeros((unique_labels.shape[0], self.k))

        l_r = np.zeros((unique_labels.shape[0]))

        for idx, l in enumerate(unique_labels):
            indices = np.where(labels == l)
            features_l = features[indices, :]
            centers[idx, :] = np.mean(features_l, axis=0)
            d[idx, :] = self.compute_top_k(features_l)
            l_r[idx] = self.k / np.sum(d[idx, :])

        l_intra = np.sum(l_r)
        d_center = self.compute_min_dist(centers)
        l_inter = max(self.margin - d_center, 0)

        self.assign(in_grad[0], req[0], mx.nd.array(y))


@mx.operator.register('rangeloss')
class RangeLossOpProp(mx.operator.CustomOpProp):
    def __init__(self, num_hidden, alpha, beta, margin=20000, k=2):
        super(RangeLossOpProp, self).__init__(need_top_grad=False)
        self.margin = margin
        self.num_hidden = int(num_hidden)
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.k = int(k)

    def list_arguments(self):
        return ['data', 'label']

    def list_outputs(self):
        return ['output']

    def infer_shape(self, in_shape):
        data_shape = in_shape[0]
        label_shape = (in_shape[0][0], )
        out_shape = in_shape[0]

        return [data_shape, label_shape], [out_shape], []

    def infer_type(self, in_type):
        dtype = in_type[0]

        return [dtype, dtype], [dtype], []

    def create_operator(self, ctx, shapes, dtypes):
        return RangeLossOp(margin=self.margin, alpha=self.alpha, beta=self.beta, k=self.k)
