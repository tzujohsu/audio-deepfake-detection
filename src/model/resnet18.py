import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, ReLU, Add, GlobalAveragePooling2D, Dense
from tensorflow.keras.models import Model

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

