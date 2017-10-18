import tensorflow as tf
from sklearn import datasets
import numpy as np
import pandas as pd


class LogisticRegression(object):
    def __init__(self, learning_rate=1e-4, epoch=10000, batch_size=50):
        self.learning_rate = learning_rate
        self.epoch = epoch
        self.batch_size = batch_size


    def train(self, train_X, train_Y, valid_X=None, valid_Y=None):
        row_num, col_num = train_X.shape
        label_num = train_Y.shape[1]
        n_batch = int(row_num / self.batch_size)

        x = tf.placeholder(dtype=tf.float32, shape=[None, col_num])
        y = tf.placeholder(dtype=tf.float32, shape=[None, label_num])

        w = tf.Variable(tf.ones([col_num, label_num]))
        b = tf.Variable(tf.ones([1, label_num]))

        y_preds = tf.nn.softmax(tf.matmul(x, w)+b)

        cross_entropy = -tf.reduce_sum(y*tf.log(y_preds))

        correct_prediction = tf.equal(tf.arg_max(y_preds, 1), tf.arg_max(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float32"))

        optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(cross_entropy)

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            old_accuracy = 0
            for i in range(self.epoch):
                if i % 10 == 0 and valid_X != None and valid_Y != None:
                    train_accuracy = sess.run(accuracy, feed_dict={x: valid_X, y: valid_Y})
                    print "step_%d, training accuracy %g" % (i, train_accuracy)

                    if abs(train_accuracy-old_accuracy) < 1e-4:
                        break
                    else:
                        old_accuracy = train_accuracy

                for n in range(n_batch):
                    train_X_batch = train_X[n * self.batch_size: (n + 1) * self.batch_size, :]
                    train_Y_batch = train_Y[n * self.batch_size: (n + 1) * self.batch_size, :]
                    sess.run(optimizer, feed_dict={x: train_X_batch, y: train_Y_batch})

            self.w, self.b = sess.run([w, b])
        return self


    def predict_proba(self, test_X):
        prob = tf.nn.softmax(tf.matmul(test_X, self.w)+self.b)
        with tf.Session() as sess:
            return sess.run(prob)


    def predict(self, test_X):
        pred = tf.arg_max(tf.matmul(test_X, self.w)+self.b, 1)
        with tf.Session() as sess:
            return sess.run(pred)


if __name__ == "__main__":
    from tensorflow.examples.tutorials.mnist import input_data

    mnist = input_data.read_data_sets('./', one_hot=True)
    train_X = mnist.train.images
    train_Y = mnist.train.labels
    test_X = mnist.test.images
    test_Y = mnist.test.labels

    model = LogisticRegression(epoch=1000, batch_size=50, learning_rate=1e-4)
    model.train(train_X, train_Y, test_X, test_Y)

