import tensorflow as tf
import tensorflow.contrib.eager as tfe
import numpy as np
from tqdm import tqdm
import os
import time


class Trainer(object):
    def __init__(self, model, data, val):
        self.model = model
        self.data = data
        self.val = val

        self.optimizer = tf.train.AdamOptimizer()

        self.summary_writer = tf.contrib.summary.create_file_writer("logs")

    def train(self, epochs, batch_size, log_per, n_train_batches, n_test_batches, save_path):
        print("Beginning training...")
        prev_val_acc = 0

        for ep in range(epochs):
            print("\nEpoch: %s" % ep)

            iterator = tfe.Iterator(self.data)
            bar = tqdm(enumerate(iterator, 1), total=n_train_batches)

            losses = []

            batch_loss = 0

            for batch, data in bar:
                imgs, labels = data

                grads, loss = self.compute_grads(imgs, labels)
                batch_loss += loss

                self.optimizer.apply_gradients(zip(grads, self.model.variables),
                                               global_step=tf.train.get_or_create_global_step())

                if batch % log_per == 0:

                    losses.append(batch_loss/log_per)  # compute avg loss across batch
                    batch_loss = 0
                    avg_loss = np.mean(losses)  #

                    bar.set_description("Loss: %.3f" % avg_loss)

            # run on val set
            val_loss, val_acc = self.eval(n_test_batches)

            if val_acc > prev_val_acc:
                # save model and continue training
                self.checkpoint(model=self.model, optimizer=self.optimizer,
                                step=tf.train.get_or_create_global_step(), path=save_path
                                )
                prev_val_acc = val_acc

            else:
                print("\nEarly stopping point reached, validation accuracy no longer improving.")
                return

        return

    def eval(self, n_batches):
        print("\nRunning eval on validation set...")
        iterator = tfe.Iterator(self.val)
        # bar = tqdm(enumerate(iterator, 1), total=n_batches)

        losses = []
        accs = []

        for data in iterator:
            imgs, labels = data

            logits, probs = self.model(imgs)

            loss = self.compute_loss(logits=logits, labels=labels)
            acc = self.compute_accuracy(tf.argmax(probs, axis=1, output_type=tf.int64), labels)

            losses.append(loss)
            accs.append(acc)

        mean_loss = np.mean(losses)
        mean_acc = np.mean(accs)

        print("Loss: %.3f Accuracy: %.3f" % (mean_loss, mean_acc))

        return mean_loss, mean_acc

    def prune_and_eval(self, n_batches, k_vals, mode="weights"):

        for k in k_vals:

            self.model.prune(k, mode)

            iterator = tfe.Iterator(self.data)
            bar = tqdm(enumerate(iterator, 1), total=n_batches)
            total_loss = 0
            total_acc = 0
            for batch, data in bar:
                imgs, labels = data
                logits, probs = self.model(imgs)
                loss = self.compute_loss(labels, logits)
                acc = self.compute_accuracy(tf.argmax(probs, axis=1, output_type=tf.int64), labels)
                total_loss += loss
                total_acc += acc

            avg_loss = total_loss / n_batches
            avg_acc = total_acc / n_batches

            print("\nEvaluating on k = %.2f...Loss: %.3f  Accuracy %.3f" % (k, avg_loss, avg_acc))

    def compute_grads(self, imgs, labels):

        with tfe.GradientTape() as tape:
            logits, _ = self.model(imgs)
            loss = self.compute_loss(labels, logits)

        return tape.gradient(loss, self.model.variables), loss

    def compute_loss(self, labels, logits):
        return tf.losses.sparse_softmax_cross_entropy(labels, logits)

    def compute_accuracy(self, preds, labels):
        return tf.reduce_mean(tf.cast(tf.equal(preds, labels), tf.float32))

    def checkpoint(self, model, optimizer, step, path):
        prefix = os.path.join(path, "ckpt")
        checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=model, optimizer_step=step)

        checkpoint.save(prefix)

    def load(self, path):
        checkpoint = tf.train.Checkpoint(optimizer=self.optimizer, model=self.model,
                                         optimizer_step=tf.train.get_or_create_global_step()
                                         )

        checkpoint.restore(tf.train.latest_checkpoint(path))


