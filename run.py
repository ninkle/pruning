import click
import os
import pickle
import tensorflow as tf
import tensorflow.contrib.eager as tfe
from pruning.model import Model
from pruning.train import Trainer
from pruning.data import MNISTLoader

os.environ['KMP_DUPLICATE_LIB_OK']='True'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

@click.group()
def cli():
    pass

@cli.command()
@click.option("--epochs", default=50)
@click.option("--batch-size", default=32)
@click.option("--log-per", default=100)
@click.option("--k-vals", default=[0.0, .25, .50, .60, .70, .80, .90, .95, .97, .99])
@click.option("--save-path", default="checkpoints")
@click.option("--log-path", default="logs")
@click.option("--results_path", default="results.pickle")
@click.option("--load-trained", is_flag=True)

def run(epochs, batch_size, log_per, k_vals, save_path, log_path, results_path, load_trained):

    tf.enable_eager_execution()

    model = Model()
    mnist = MNISTLoader()
    n_train_batches, train = mnist.load_train(batch_size)
    n_test_batches, test = mnist.load_test(batch_size)
    trainer = Trainer(model, data=train, val=test, save_path=save_path, log_path=log_path)

    if load_trained:
        checkpoint = tf.train.Checkpoint(optimizer=trainer.optimizer, model=trainer.model)
        trainer.load(checkpoint, save_path)
        val_loss, val_acc = trainer.eval()

    else: # train model
        _, _, val_acc, val_loss = trainer.train(epochs, log_per, n_train_batches, n_test_batches)

    # prune on the weight level
    weight_losses, weight_accs = trainer.prune_and_eval(n_test_batches, k_vals, mode="weights")

    # prune on the unit level
    unit_losses, unit_accs = trainer.prune_and_eval(n_test_batches, k_vals, mode="units")

    # store and serialize results
    results = {"weight_losses": weight_losses, "weight_accs": weight_accs,
               "unit_losses": unit_losses, "unit_accs": unit_accs,
               "val_acc": val_acc, "val_loss": val_loss}

    with open(results_path, "wb") as f:
        pickle.dump(results, f)

if __name__ == "__main__":
    cli()