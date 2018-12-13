import tensorflow as tf
import numpy as np
tf.enable_eager_execution()

class Model(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.hidden1 = DenseLayer(1000)
        self.hidden2 = DenseLayer(1000)
        self.hidden3 = DenseLayer(500)
        self.hidden4 = DenseLayer(200)
        self.output_layer = DenseLayer(10)

        self.net = tf.keras.Sequential([self.hidden1, self.hidden2, self.hidden3, self.hidden4, self.output_layer])

    def __call__(self, x, training=True):
        h1 = self.hidden1(x)
        h2 = self.hidden2(tf.nn.relu(h1))
        h3 = self.hidden3(tf.nn.relu(h2))
        h4 = self.hidden4(tf.nn.relu(h3))

        logits = self.output_layer(tf.nn.relu(h4))
        probs = tf.nn.softmax(logits, -1)

        return logits, probs

    def prune(self, k=0.1, mode="weights"):
        """
        :param mode: "weights" or "units"
        :return:
        """
        pruned = []
        if mode == "weights":
            print("\nPruning %.2f percent of weights..." % k)
            for layer in self.net.get_weights()[:-1]:  # ignore output layer
                shape = layer.shape

                flattened = tf.reshape(layer, (1, -1))
                norm_flattened = tf.math.abs(flattened)

                idx = int(int(norm_flattened.shape[-1]) * k)
                val = tf.contrib.framework.sort(norm_flattened)[:, idx]

                mask = tf.cast(norm_flattened > val, tf.float32)
                masked_flattened = tf.multiply(flattened, mask)
                masked_matrix = tf.reshape(masked_flattened, shape)

                pruned.append(masked_matrix)

        elif mode == "units":
            print("\nPruning %.2f percent of units..." % k)
            for layer in self.net.get_weights()[:-1]:
                norm = tf.norm(layer, ord=2, axis=-1)

                idx = int(int(norm.shape[-1]) * k)
                val = tf.contrib.framework.sort(norm)[idx]

                masked = tf.cast(layer > val, tf.float32)

                pruned.append(masked)

        self.net.set_weights(pruned)


class DenseLayer(tf.keras.layers.Layer):
    def __init__(self, output_units):
        super(DenseLayer, self).__init__()
        self.output_units = output_units

    def build(self, input_shape):
        self.w = self.add_variable("weights", shape=(input_shape[-1].value, self.output_units))

    def call(self, input):
        return tf.matmul(input, self.w)

