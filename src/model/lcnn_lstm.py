# pylint: disable=E0402,W0622
from typing import List

import tensorflow as tf
from keras.layers import Layer, BatchNormalization, Dense, Dropout, Flatten, Input, MaxPool2D, Permute, Multiply, Lambda, GlobalAveragePooling1D, TimeDistributed
from keras.layers import Conv2D, LSTM
from keras.models import Model
from tensorflow.keras.layers import Reshape
from tensorflow.keras import backend as K

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

class SelfAttentivePooling(Layer):
    def __init__(self, **kwargs):
        super(SelfAttentivePooling, self).__init__(**kwargs)

    def build(self, input_shape):
        self.attention_dense = Dense(1, activation='tanh')
        super(SelfAttentivePooling, self).build(input_shape)

    def call(self, x):
        # Compute attention scores
        u = self.attention_dense(x)
        u = Permute((2, 1))(u)
        u = Lambda(lambda x: K.softmax(x, axis=2))(u)
        u = Permute((2, 1))(u)

        # Apply attention scores
        out = Multiply()([x, u])
        out = Lambda(lambda x: K.sum(x, axis=1))(out)
        return out


def build_lcnn_lstm(shape: List[int], n_label: int = 2) -> tf.keras.Model:
    input = Input(shape=shape)

    # LCNN Layers
    x = MaxOutConv2D(input, 64, kernel_size=5, strides=1, padding="same")
    x = MaxPool2D(pool_size=(2, 2))(x)

    x = MaxOutConv2D(x, 64, kernel_size=1, strides=1, padding="same")
    x = BatchNormalization()(x)

    x = MaxOutConv2D(x, 96, kernel_size=3, strides=1, padding="same")
    x = MaxPool2D(pool_size=(2, 2))(x)
    x = BatchNormalization()(x)

    x = MaxOutConv2D(x, 128, kernel_size=3, strides=1, padding="same")
    x = MaxPool2D(pool_size=(2, 2))(x)

    # Flatten and Reshape into sequence
    x = TimeDistributed(Flatten())(x)  # Keeps time dimension intact
    x = LSTM(64, return_sequences=True)(x)

    # Self-Attentive Pooling
    x = SelfAttentivePooling()(x)

    # Final Dense Layer & Output
    x = MaxOutDense(x, 64)
    x = BatchNormalization()(x)
    output = Dense(n_label, activation="sigmoid" if n_label == 1 else "softmax")(x)


    return Model(inputs=input, outputs=output)
