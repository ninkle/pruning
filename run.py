import click
import tensorflow as tf
import tensorflow.contrib.eager as tfe
from pruning.model import Model
from pruning.train import Trainer

@click.group()
def cli():
    pass

@cli.command()
@click.option("--epochs", default=50)
@click.option("--batch-size", default=32)
@click.option("--log-per", default=100)
@click.option("--k-vals", default=[0.0, .25, .50, .60, .70, .80, .90, .95, .97, .99])
@click.option("--save-path", default="trained")


def run(epochs, batch_size, log_per, k_vals, save_path):

    tfe.enable_eager_execution()

    # load mnist
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    x_train_flattened = tf.reshape(x_train, shape=(-1, x_train.shape[1] * x_train.shape[2]))
    x_test_flattened = tf.reshape(x_test, shape=(-1, x_test.shape[1] * x_test.shape[2]))

    x_train_dataset = tf.data.Dataset.from_tensor_slices(tf.cast(x_train_flattened, tf.float32))
    x_test_dataset = tf.data.Dataset.from_tensor_slices(tf.cast(x_test_flattened, tf.float32))
    y_train_dataset = tf.data.Dataset.from_tensor_slices(tf.cast(y_train, tf.int64))
    y_test_dataset = tf.data.Dataset.from_tensor_slices(tf.cast(y_test, tf.int64))

    train = tf.data.Dataset.zip((x_train_dataset, y_train_dataset))
    test = tf.data.Dataset.zip((x_test_dataset, y_test_dataset))

    train = train.batch(batch_size)
    test = test.batch(batch_size)

    n_batches_train = int(x_train.shape[0] / batch_size)
    n_batches_test = int(x_test.shape[0] / batch_size)

    model = Model()
    trainer = Trainer(model, data=train, val=test)

    trainer.train(epochs, batch_size, log_per, n_batches_train, n_batches_test, save_path)
    trainer.prune_and_eval(n_batches_train, k_vals, mode="weights")
    trainer.prune_and_eval(n_batches_train, k_vals, mode="units")

if __name__ == "__main__":
    cli()