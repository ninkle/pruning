import tensorflow as tf
import tensorflow.contrib.eager as tfe
import numpy as np
from tqdm import tqdm
import os

class Trainer(object):
    def __init__(self, model, data, val, save_path, log_path):
        self.model = model
        self.data = data
        self.val = val

        self.optimizer = tf.train.AdamOptimizer()

        self.save_path = save_path
        self.log_path = log_path
        self.summary_writer = tf.contrib.summary.create_file_writer(log_path, flush_millis=10000)

    def train(self, epochs, log_per, n_train_batches, n_test_batches):
        print("Beginning training...")
        global_step = tf.train.get_or_create_global_step()
        checkpoint = tf.train.Checkpoint(optimizer=self.optimizer, model=self.model, global_step=global_step)
        prev_val_acc = 0

        with self.summary_writer.as_default(), tf.contrib.summary.always_record_summaries():
            for ep in range(epochs):
                print("\nEpoch: %s" % ep)

                iterator = tfe.Iterator(self.data)
                bar = tqdm(enumerate(iterator, 1), total=n_train_batches)

                losses = []

                batch_loss = 0

                for batch, data in bar:
                    imgs, labels = data

                    logits, probs = self.model(imgs)
                    grads, loss = self.compute_grads(imgs, labels)
                    batch_loss += loss

                    self.optimizer.apply_gradients(zip(grads, self.model.variables),
                                                   global_step=global_step)

                    if batch % log_per == 0:

                        losses.append(batch_loss/log_per)  # compute avg loss across batch

                        batch_loss = 0
                        avg_loss = np.mean(losses)

                        tf.contrib.summary.scalar("train_loss", avg_loss)

                        bar.set_description("Loss: %.3f" % avg_loss)

                # run on val set
                val_loss, val_acc = self.eval(n_test_batches)
                tf.contrib.summary.scalar("val_loss", val_loss)
                tf.contrib.summary.scalar("val_acc", val_acc)

                if val_acc > prev_val_acc:
                    # save model and continue training
                    self.save(checkpoint, self.save_path)
                    prev_val_acc = val_acc
                    prev_val_loss = val_loss

                else:
                    print("\nEarly stopping point reached, validation accuracy no longer improving.")
                    train_loss, train_acc = self.eval(data_src="train")
                    return train_loss, train_acc, prev_val_acc, prev_val_loss

            return

    def eval(self, data_src="val"):
        if data_src is "train":
            data = self.data
        else:
            data = self.val

        print("\nRunning eval on validation set...")
        iterator = tfe.Iterator(data)

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

        losses = []
        accs = []
        checkpoint = tf.train.Checkpoint(model=self.model)

        for k in k_vals:

            checkpoint.restore(tf.train.latest_checkpoint(self.save_path))

            self.model.prune(k, mode)

            iterator = tfe.Iterator(self.val)
            bar = tqdm(enumerate(iterator, 1), total=n_batches)

            k_losses = []
            k_accs = []

            for batch, data in bar:
                imgs, labels = data
                logits, probs = self.model(imgs)
                loss = self.compute_loss(labels, logits)
                acc = self.compute_accuracy(tf.argmax(probs, axis=1, output_type=tf.int64), labels)
                k_losses.append(loss)
                k_accs.append(acc)

            avg_k_loss = np.mean(k_losses)
            avg_k_acc = np.mean(k_accs)

            losses.append(avg_k_loss)
            accs.append(avg_k_acc)

            print("\nEvaluating on k = %.2f...Loss: %.3f  Accuracy %.3f" % (k, avg_k_loss, avg_k_acc))

        return losses, accs

    def compute_grads(self, imgs, labels):

        with tfe.GradientTape() as tape:
            logits, _ = self.model(imgs)
            loss = self.compute_loss(labels, logits)

        return tape.gradient(loss, self.model.variables), loss

    def compute_loss(self, labels, logits):
        return tf.losses.sparse_softmax_cross_entropy(labels, logits)

    def compute_accuracy(self, preds, labels):
        return tf.reduce_mean(tf.cast(tf.equal(preds, labels), tf.float32))

    def save(self, checkpoint, path):
        prefix = os.path.join(path, "ckpt")

        checkpoint.save(prefix)

    def load(self, checkpoint, path):
        checkpoint.restore(tf.train.latest_checkpoint(path))


