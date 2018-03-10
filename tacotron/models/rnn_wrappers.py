"""A set of RNN wrappers usefull for tacotron 2 architecture
All notations and variable names were used in concordance with originial tensorflow implementation
Some tensors were passed through wrappers to make sure we respect the described architecture
"""

import numpy as np
import tensorflow as tf
from tensorflow.contrib.rnn import RNNCell
from .modules import prenet, projection
from tensorflow.python.framework import ops
from hparams import hparams



class DecoderPrenetWrapper(RNNCell):
  '''Runs RNN inputs through a prenet before sending them to the cell.'''
  def __init__(self, cell, is_training):
    super(DecoderPrenetWrapper, self).__init__()
    self._cell = cell
    self._is_training = is_training

  @property
  def state_size(self):
    return self._cell.state_size

  @property
  def output_size(self):
    return self._cell.output_size

  def call(self, inputs, state):
    prenet_out = prenet(inputs, self._is_training, hparams.prenet_layers, scope='decoder_attention_prenet')
    self._prenet_out = prenet_out
    return self._cell(prenet_out, state)

  def zero_state(self, batch_size, dtype):
    return self._cell.zero_state(batch_size, dtype)


class ConcatPrenetAndAttentionWrapper(RNNCell):
  '''Concatenates prenet output with the attention context vector.
  This is expected to wrap a cell wrapped with an AttentionWrapper constructed with
  attention_layer_size=None and output_attention=False. Such a cell's state will include an
  "attention" field that is the context vector.
  '''
  def __init__(self, cell):
    super(ConcatPrenetAndAttentionWrapper, self).__init__()
    self._cell = cell

  @property
  def state_size(self):
    return self._cell.state_size

  @property
  def output_size(self):
    #attention is stored in attentionwrapper cell state
    return self._cell.output_size + self._cell.state_size.attention

  def call(self, inputs, state):
    #We assume paper writers mentionned the attention network output when
    #they say "The pre-net output and attention context vector are concatenated and
    #passed through a stack of 2 uni-directional LSTM layers"
    #We rely on the original tacotron architecture for this hypothesis.
    output, res_state = self._cell(inputs, state)

    #Store attention in this wrapper to make access easier from future wrappers
    self._context_vector = res_state.attention
    return tf.concat([output, self._context_vector], axis=-1), res_state

  def zero_state(self, batch_size, dtype):
    return self._cell.zero_state(batch_size, dtype)


class ConcatLSTMOutputAndAttentionWrapper(RNNCell):
  '''Concatenates decoder RNN cell output with the attention context vector.
  This is expected to wrap a cell wrapped with an AttentionWrapper constructed with
  attention_layer_size=None and output_attention=False. Such a cell's state will include an
  "attention" field that is the context vector.
  '''
  def __init__(self, cell):
    super(ConcatLSTMOutputAndAttentionWrapper, self).__init__()
    self._cell = cell
    self._prenet_attention_cell = self._cell._cells[0]

  @property
  def state_size(self):
    return self._cell.state_size

  @property
  def output_size(self):
    return self._cell.output_size + self._prenet_attention_cell.state_size.attention

  def call(self, inputs, state):
    output, res_state = self._cell(inputs, state)
    context_vector = self._prenet_attention_cell._context_vector
    self.lstm_concat_context = tf.concat([output, context_vector], axis=-1)
    return self.lstm_concat_context, res_state

  def zero_state(self, batch_size, dtype):
    return self._cell.zero_state(batch_size, dtype)
