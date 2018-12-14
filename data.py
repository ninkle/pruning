import tensorflow as tf

class MNISTLoader(object):
    def __init__(self):
        (self.x_train, self.y_train), (self.x_test, self.y_test) = tf.keras.datasets.mnist.load_data()

        x_train_flattened = tf.reshape(self.x_train, shape=(-1, self.x_train.shape[1] * self.x_train.shape[2]))
        x_test_flattened = tf.reshape(self.x_test, shape=(-1, self.x_test.shape[1] * self.x_test.shape[2]))

        x_train_dataset = tf.data.Dataset.from_tensor_slices(tf.cast(x_train_flattened, tf.float32))
        x_test_dataset = tf.data.Dataset.from_tensor_slices(tf.cast(x_test_flattened, tf.float32))
        y_train_dataset = tf.data.Dataset.from_tensor_slices(tf.cast(self.y_train, tf.int64))
        y_test_dataset = tf.data.Dataset.from_tensor_slices(tf.cast(self.y_test, tf.int64))

        self.train = tf.data.Dataset.zip((x_train_dataset, y_train_dataset))
        self.train_size = self.x_train.shape[0]
        self.batched_train = self.train.batch
        self.test = tf.data.Dataset.zip((x_test_dataset, y_test_dataset))
        self.test_size = self.y_test.shape[0]

    def load_train(self, batch_size=None):
        if batch_size is not None:
            n_batches = int(self.train_size / batch_size)
            batched = self.train.batch(batch_size)

        else:
            n_batches = self.train_size
            batched = self.train.batch(1)

        return n_batches, batched

    def load_test(self, batch_size=None):
        if batch_size is not None:
            n_batches = int(self.test_size / batch_size)
            batched = self.test.batch(batch_size)

        else:
            n_batches = self.test_size
            batched = self.test.batch(1)

        return n_batches, batched


