import tensorflow as tf
import numpy as np



# linear regression in tensorflow

class LinearRegression(object):

    def __init__(self, learning_rate=1e-4, epoch=10000, batch_size=50):
        self.learning_rate = learning_rate
        self.epoch = epoch
        self.batch_size = batch_size


    def train(self, train_X, train_Y):
        # y must be column

        row_num, col_num = train_X.shape

        x = tf.placeholder(dtype=tf.float32, shape=[None, col_num])
        y = tf.placeholder(dtype=tf.float32, shape=[None, 1])

        w = tf.Variable(tf.truncated_normal([col_num, 1]), name="weight")
        b = tf.Variable(tf.truncated_normal([1, 1]), name="bias")

        y_predict = tf.matmul(x, w)+b

        loss = tf.reduce_mean(tf.pow(y_predict-y, 2))

        optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(loss)

        n_batch = int(row_num / self.batch_size)

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            for _ in range(self.epoch):
                for n in range(n_batch):
                    sess.run(
                        optimizer,
                        feed_dict={x: train_X[n * self.batch_size:(n+1) * self.batch_size, :],
                                   y: train_Y[n * self.batch_size:(n+1) * self.batch_size, :]}
                        )

            self.w, self.b = sess.run([w, b])
        return self

    def predict(self, test_X):
        pred = tf.matmul(test_X, self.w)+self.b
        with tf.Session() as sess:
            preds = sess.run(pred)
        return preds




if __name__=="__main__":
    from sklearn import datasets

    boston = datasets.load_iris()
    train_X = np.mat(boston.data)
    train_Y = np.mat(boston.target).T
    model = LinearRegression(batch_size=10)
    model.train(train_X, train_Y)
    print model.w, model.b

