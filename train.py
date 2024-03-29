#!/usr/bin/python3

from os.path import join, exists
from absl import flags, app
import tensorflow as tf
import tensorflow_gnn as tfgnn
from create_datasets import get_parse_function
from models import AEROGNN

FLAGS = flags.FLAGS

def add_options():
  flags.DEFINE_integer('head', default = 1, help = 'number of head')
  flags.DEFINE_integer('channels', default = 64, help = 'channel size')
  flags.DEFINE_string('dataset', default = None, help = 'path to directory containing train/val set')
  flags.DEFINE_integer('batch', default = 256, help = 'batch size')
  flags.DEFINE_float('lr', default = 1e-3, help = 'learning rate')
  flags.DEFINE_string('ckpt', default = 'ckpt', help = 'checkpoint path')
  flags.DEFINE_integer('decay_steps', default = 10000, help = 'decay steps')
  flags.DEFINE_integer('save_freq', default = 1000, help = 'checkpoint save frequency')
  flags.DEFINE_integer('epochs', default = 200, help = 'epochs to run')

def main(unused_argv):
  trainset = tf.data.TFRecordDataset(join(FLAGS.dataset, 'trainset.tfrecord')).map(get_parse_function(head = FLAGS.head, channels = FLAGS.channels)).prefetch(FLAGS.batch).shuffle(FLAGS.batch).batch(FLAGS.batch)
  valset = tf.data.TFRecordDataset(join(FLAGS.dataset, 'testset.tfrecord')).map(get_parse_function(head = FLAGS.head, channels = FLAGS.channels)).prefetch(FLAGS.batch).shuffle(FLAGS.batch).batch(FLAGS.batch)
  model = AEROGNN(head = FLAGS.head, channels = FLAGS.channels)
  loss = [tf.keras.losses.MeanAbsoluteError()]
  metrics = [tf.keras.metrics.MeanAbsoluteError()]
  optimizer = tf.keras.optimizers.Adam(tf.keras.optimizers.schedules.CosineDecayRestarts(FLAGS.lr, first_decay_steps = FLAGS.decay_steps))
  model.compile(optimizer = optimizer, loss = loss, metrics = metrics)
  if exists(FLAGS.ckpt): model.load_weights(join(FLAGS.ckpt, 'ckpt', 'variables', 'variables'))
  callbacks = [
    tf.keras.callbacks.TensorBoard(log_dir = FLAGS.ckpt),
    tf.keras.callbacks.ModelCheckpoint(filepath = join(FLAGS.ckpt, 'ckpt'), save_freq = FLAGS.save_freq, save_best_only = True, mode = "min")]
  model.fit(trainset, epochs = FLAGS.epochs, validation_data = valset, callbacks = callbacks)

if __name__ == "__main__":
  add_options()
  app.run(main)

