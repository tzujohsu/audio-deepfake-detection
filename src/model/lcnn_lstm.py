# pylint: disable=E0402,W0622
from typing import List

import tensorflow as tf
from keras.layers import BatchNormalization, Dense, Dropout, Flatten, Input, MaxPool2D
from keras.layers import Conv2D, LSTM
from keras.models import Model
from tensorflow.keras.layers import Reshape

from .layers import Maxout


# function that return the stuck of Conv2D and MFM
def MaxOutConv2D(input: tf.Tensor, dim: int, kernel_size: int, strides: int, padding: str = "same") -> tf.Tensor:
    """MaxOutConv2D

    This is a helper function for LCNN class.
    This function combine Conv2D layer and Mac Feature Mapping function (MFM).
    Makes codes more readable.

    Args:
      input(tf.Tensor): The tensor from a previous layer.
      dim(int): Dimenstion of the Convolutional layer.
      kernel_size(int): Kernel size of Convolutional layer.
      strides(int): Strides for Convolutional layer.
      padding(string): Padding for Convolutional layer, "same" or "valid".

     Returns:
      mfm_out: Outputs after MFM.

    Examples:
      conv2d_1 = MaxOutConv2D(input, 64, kernel_size=2, strides=2, padding="same")

    """
    conv_out = Conv2D(dim, kernel_size=kernel_size, strides=strides, padding=padding)(input)
    mfm_out = Maxout(int(dim / 2))(conv_out)
    return mfm_out


# function that return the stuck of FC and MFM
def MaxOutDense(x: tf.Tensor, dim: int) -> tf.Tensor:
    """MaxOutDense

    Almost same as MaxOutConv2D.
    Only the difference is that the layer before mfm is Dense layer.

    """
    dense_out = Dense(dim)(x)
    mfm_out = Maxout(int(dim / 2))(dense_out)
    return mfm_out



def build_lcnn_lstm(shape: List[int], n_label: int = 2) -> tf.keras.Model:
    input = Input(shape=shape)

    # First Conv Block
    conv2d_1 = MaxOutConv2D(input, 64, kernel_size=5, strides=1, padding="same")
    maxpool_1 = MaxPool2D(pool_size=(2, 2), strides=(2, 2))(conv2d_1)

    # Second Conv Block
    conv2d_2 = MaxOutConv2D(maxpool_1, 96, kernel_size=3, strides=1, padding="same")
    maxpool_2 = MaxPool2D(pool_size=(2, 2), strides=(2, 2))(conv2d_2)
    batch_norm_2 = BatchNormalization()(maxpool_2)

    # Third Conv Block
    conv2d_3 = MaxOutConv2D(batch_norm_2, 64, kernel_size=3, strides=1, padding="same")
    maxpool_3 = MaxPool2D(pool_size=(2, 2), strides=(2, 2))(conv2d_3)
    batch_norm_3 = BatchNormalization()(maxpool_3)

    # Fourth Conv Block
    conv2d_4 = MaxOutConv2D(batch_norm_3, 64, kernel_size=3, strides=1, padding="same")
    maxpool_4 = MaxPool2D(pool_size=(2, 2), strides=(2, 2))(conv2d_4)
    flatten = Flatten()(maxpool_4)

    # Dense Layer
    dense = MaxOutDense(flatten, 128)
    # dropout = Dropout(0.25)(dense)

    # Reshape for LSTM
    reshape = Reshape((1, 64))(dense)

    # LSTM Layer
    lstm = LSTM(64, return_sequences=False)(reshape)
    # batch_norm_lstm = BatchNormalization()(lstm)
    dense_lstm = MaxOutDense(lstm, 64)
    batch_norm_lstm = BatchNormalization()(dense_lstm)
    
    

    # Output Layer
    output = Dense(n_label, activation="softmax")(batch_norm_lstm)

    return Model(inputs=input, outputs=output)
