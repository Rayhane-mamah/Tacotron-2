import numpy as np
import tensorflow as tf
from tensorflow.contrib.rnn import RNNCell
from .modules import prenet, projection
from hparams import hparams


class TacotronDecoderWrapper(RNNCell):
  """Computes custom Tacotron decoder and return decoder output and state at each step
  
  decoder architecture:
    Prenet: 2 dense layers, 128 units each
      * concat(Prenet output + context vector) 
    RNNStack (LSTM): 2 uni-directional LSTM layers with 512 units each
      * concat(LSTM output + context vector)
    Linear projection layer: output_dim = decoder_output
  """
  def __init__(self, cell, is_training):
    super(TacotronDecoderWrapper, self).__init__()
    self._cell = cell
    self._is_training = is_training

  @property
  def state_size(self):
    return self._cell.state_size

  @property
  def output_size(self):
    #return (self.batch_size, hparams.num_mels)
    return self._cell.output_size

  def call(self, inputs, state):
    #Get context vector from cell state
    context_vector = state.attention
    cell_state = state.cell_state

    #Compute prenet output
    prenet_outputs = prenet(inputs, self._is_training, scope='decoder_prenet_layer')

    #Concat prenet output and context vector
    concat_output_prenet = tf.concat([prenet_outputs, context_vector], axis=-1)

    #Compute LSTM output
    LSTM_output, next_cell_state = self._cell(concat_output_prenet, cell_state)

    #Concat LSTM output and context vector
    concat_output_LSTM = tf.concat([LSTM_output, context_vector], axis=-1)

    #Linear projection
    proj_shape = hparams.num_mels
    cell_output = (projection(concat_output_LSTM, proj_shape, scope='decoder_projection_layer'), LSTM_output)

    return cell_output, next_cell_state

  def zero_state(self, batch_size, dtype):
    return self._cell.zero_state(batch_size, dtype)
