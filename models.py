#!/usr/bin/python3

import tensorflow as tf
import tensorflow_gnn as tfgnn
from create_datasets import graph_tensor_spec

def FeatInit(hid_channels = 64, dense_layer_num = 2, drop_rate = 0.5):
  model = tf.keras.Sequential()
  for i in range(dense_layer_num):
    if i != 0: model.add(tf.keras.layers.ELU())
    model.add(tf.keras.layers.Dropout(rate = drop_rate))
    model.add(tf.keras.Dense(hid_channels))
  return model

class UpdateZ(tf.keras.layers.Layer):
  def __init__(self, k, **kwargs):
    super(UpdateZ, self).__init__()
    self.channels = kwargs.get('channels', 64)
    self.head = kwargs.get('head', 1)
    self.lambd = kwargs.get('lambd', 1.)
    self.k = k
  def build(self, input_shape):
    self.hop_att = self.add_weight(name = 'hop_att', shape = (1, self.head, (self.channels // self.head) if self.k == 0 else (self.channels // self.head * 2)), trainable = True)
    self.hop_bias = self.add_weight(name = 'hop_bias', shape = (1, self.head), trainable = True)
  def call(self, graph):
    h = tfgnn.keras.layers.Readout(node_set_name = 'atom', feature_name = tfgnn.HIDDEN_STATE)(graph) # hidden.shape = (node_num, channels)
    if self.k != 0:
      # NOTE: get previous calculated hidden
      z = tfgnn.keras.layers.Readout(from_context = True, feature_name = tfgnn.HIDDEN_STATE)(graph) # z.shape = (node_num, head, channels // head)
      z_scale = z * tf.math.log((self.lambd / self.k) + (1 + 1e-6))
      h = tf.concat([h, z_scale], axis = -1) # h.shape = (node_num, head, channels // head * 2)
    h = tf.reshape(h, (-1, self.head, (self.channels // self.head) if self.k == 0 else (self.channels // self.head * 2))) # hidden.shape = (node_num, head, channels // head)
    hop_attention = tf.nn.elu(h) # hop_attention.shape = (node_num, head, channels // head)
    hop_attention = tf.math.reduce_sum(self.hop_att * hop_attention, axis = -1) + self.hop_bias # hop_attention.shape = (node_num, head)
    if self.k == 0:
      z = hidden * hop_attention # z.shape = (node_num, head, channel // head)
    else:
      # NOTE: accumulate with previous calculated hidden
      z = z + hidden * hop_attention # z.shape = (node_num, head, channel // head)
    return z
  def get_config(self):
    config = super(UpdateZ, self).get_config()
    config['channels'] = self.channels
    config['head'] = self.head
    config['lambd'] = self.lambd
    config['k'] = k
    return config
  @classmethod
  def from_config(cls, config):
    return cls(**config)

class Propagate(tf.keras.layers.Layer):
  def __init__(self, k, **kwargs):
    super(Propagate, self).__init__()
    self.channels = kwargs.get('channels', 64)
    self.head = kwargs.get('head', 1)
    self.lambd = kwargs.get('lambd', 1.)
    self.k = k
  def build(self, input_shape):
    self.att = self.add_weight(name = 'att', shape = (1, self.head, self.channels // self.head), trainable = True)
  def call(self, graph, edge_set_name):
    z = tfgnn.keras.layers.Readout(from_context = True, feature_name = tfgnn.HIDDEN_STATE)(graph)
    z_scale = z * tf.math.log((self.lambd / self.k) + (1 + 1e-6)) # z_scale.shape = (node_num, head, channel // head)
    z_scale_i = tfgnn.broadcast_node_to_edges(graph, edge_set_name, tfgnn.SOURCE, feature_value = z_scale) # z_scale_i.shape = (edge_num, head, channel // head)
    z_scale_j = tfgnn.broadcast_node_to_edges(graph, edge_set_name, tfgnn.TARGET, feature_value = z_scale) # z_scale_j.shape = (edge_num, head, channel // head)
    a_ij = z_scale_i + z_scale_j
    a_ij = tf.nn.elu(a_ij) # a_ij.shape = (edge_num, head, channel // head)
    a_ij = tf.math.reduce_sum(self.att * a_ij, axis = -1) # a_ij.shape = (edge_num, head)
    a_ij = tf.math.softplus(a_ij) + 1e-6
    # TODO

def AEROGNN(channels = 64, head = 1, lambd = 1., layer_num = 10, drop_rate = 0.6):
  inputs = tf.keras.Input(type_spec = graph_tensor_spec())
  results = inputs.merge_batch_to_components()
  results = tfgnn.keras.layers.MapFeatures(
    node_sets_fn = lambda node_set, *, node_set_name: FeatInit(hid_channel = channels // head, drop_rate = drop_rate)(node_set[tfgnn.HIDDEN_STATE]))(results)
  # initialize z
  results = tfgnn.keras.layers.GraphUpdate(context = UpdateZ(k = 0, channels = channels, head = head, lambd = lambd))(results)
  for i in range(layer_num):
    results = tfgnn.keras.layers.GraphUpdate(
      node_sets = {
        "atom": tfgnn.keras.layers.NodeSetUpdate(
          edge_set_inputs = {
            "bond": tfgnn.keras.layers.})})
