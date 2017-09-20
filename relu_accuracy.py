from keras.datasets import mnist
import matplotlib.pyplot as plt
import numpy as np
from keras.utils import np_utils

seed = 7
np.random.seed(seed)


def softmax(z):
    return np.exp(z) / np.sum(np.exp(z))


def reverse_softmax(z):
    return softmax(z) * (1 - softmax(z))


class Network:
    def __init__(self):
        self.epsilon = 1e-0
        self.regularization = 0.001
        self.epoch = 500
        self.load_data()
        self.layer_size = [self.num_pixels, 32, self.num_classes]
        print "param : num_train, num_classes, layer_size"
        print self.num_train, self.num_classes, self.layer_size
        self.random_init()

    def load_data(self):
        (X_train, y_train), (X_test, y_test) = mnist.load_data()
        self.num_train = X_train.shape[0]
        self.num_test = X_test.shape[0]

        self.num_pixels = X_train.shape[1] * X_train.shape[2]
        X_train = X_train.reshape(X_train.shape[0], self.num_pixels).astype('float32')
        X_test = X_test.reshape((X_test.shape[0], self.num_pixels)).astype('float32')

        self.X_train = X_train / 255.0
        self.X_test = X_test / 255.0

        # self.y_train = np_utils.to_categorical(y_train)
        # self.y_test = np_utils.to_categorical(y_test)
        self.y_train = y_train
        self.y_test = y_test
        self.num_classes = 10#y_test.shape[1]
        print "param shape : X_train y_train"
        print X_train.shape, y_train.shape

    def random_init(self):
        self.w1 = 0.01 * np.random.randn(self.layer_size[0], self.layer_size[1])
        self.b1 = np.zeros((1, self.layer_size[1]))
        self.w2 = 0.01 * np.random.randn(self.layer_size[1], self.layer_size[2])
        self.b2 = np.zeros((1, self.layer_size[2]))
        print "param shape: w1, b1, w2, b2"
        print self.w1.shape, self.b1.shape, self.w2.shape, self.b2.shape
        # print np.min(self.y_train), np.max(self.y_train)

    def print_test_metrics(self):
        z1 = self.X_test.dot(self.w1) + self.b1
        a1 = np.maximum(z1, 0)  # relu
        z2 = a1.dot(self.w2) + self.b2
        exp_scores = np.exp(z2)
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        corect_logprobs = -np.log(probs[range(self.num_test), self.y_test])
        data_loss = np.sum(corect_logprobs) / self.num_test
        data_loss += self.regularization/ 2 * (np.sum(np.square(self.w1)) + np.sum(np.square(self.w2)))


        print "test loss : ", data_loss
        predicted_class = np.argmax(z2, axis=1)
        print 'test accuracy: %.2f' % (np.mean(predicted_class == self.y_test))


    def train(self):
        for i in range(self.epoch):
            print "epoch %i" % i

            # print "x ", self.X_train.shape
            # forward propagation
            z1 = self.X_train.dot(self.w1) + self.b1
            # print "z1 ", z1.shape
            a1 = np.maximum(z1, 0)  # relu
            z2 = a1.dot(self.w2) + self.b2
            # print "z2 ", z2.shape
            exp_scores = np.exp(z2)
            probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

            # print probs.shape, probs
            # print "finished: forward"

            # loss
            corect_logprobs = -np.log(probs[range(self.num_train), self.y_train])
            data_loss = np.sum(corect_logprobs) / self.num_train
            data_loss += self.regularization / 2 * (np.sum(np.square(self.w1)) + np.sum(np.square(self.w2)))
            print "data loss ", (i, data_loss)

            if i % 10 == 0 :
                predicted_class = np.argmax(z2, axis=1)
                print 'training accuracy: %.2f' % (np.mean(predicted_class == self.y_train))
                self.print_test_metrics()

            # print "finished: loss"
            # backward propagation
            d3 = probs
            d3[range(self.num_train), self.y_train] -= 1
            d3 /= self.num_train

            dw2 = (a1.T).dot(d3)
            db2 = np.sum(d3, axis=0, keepdims=True)
            d2 = np.dot(d3, self.w2.T)
            d2[a1 <= 0] = 0

            dw1 = np.dot(self.X_train.T, d2)
            db1 = np.sum(d2, axis=0, keepdims=True)

            # print "finished: backward"
            dw2 += self.regularization * self.w2
            dw1 += self.regularization * self.w1

            self.w1 += -self.epsilon*dw1
            self.b1 += -self.epsilon*db1
            self.w2 += -self.epsilon*dw2
            self.b2 += -self.epsilon*db2
            # print "finished: param update"

            # print self.calc_loss()

n = Network()
n.train()
