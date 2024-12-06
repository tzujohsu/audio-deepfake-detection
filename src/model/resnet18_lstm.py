from keras.layers import Dense, Conv2D,  MaxPool2D, Flatten, GlobalAveragePooling2D,  BatchNormalization, Layer, Add, GlobalMaxPooling2D
from keras.layers import Layer, BatchNormalization, Dense, Dropout, Flatten, Input, MaxPool2D, Permute, Multiply, Lambda, GlobalAveragePooling1D, TimeDistributed
from keras.models import Model
import tensorflow as tf
from keras.layers import Conv2D, LSTM
from tensorflow.keras import backend as K


from typing import List

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

class ResnetBlock(Model):
    """
    A standard resnet block.
    """

    def __init__(self, channels: int, down_sample=False):
        """
        channels: same as number of convolution kernels
        """
        super().__init__()

        self.__channels = channels
        self.__down_sample = down_sample
        self.__strides = [2, 1] if down_sample else [1, 1]

        KERNEL_SIZE = (3, 3)
        # use He initialization, instead of Xavier (a.k.a 'glorot_uniform' in Keras), as suggested in [2]
        INIT_SCHEME = "he_normal"

        self.conv_1 = Conv2D(self.__channels, strides=self.__strides[0],
                             kernel_size=KERNEL_SIZE, padding="same", kernel_initializer=INIT_SCHEME)
        self.bn_1 = BatchNormalization()
        self.conv_2 = Conv2D(self.__channels, strides=self.__strides[1],
                             kernel_size=KERNEL_SIZE, padding="same", kernel_initializer=INIT_SCHEME)
        self.bn_2 = BatchNormalization()
        self.merge = Add()

        if self.__down_sample:
            # perform down sampling using stride of 2, according to [1].
            self.res_conv = Conv2D(
                self.__channels, strides=2, kernel_size=(1, 1), kernel_initializer=INIT_SCHEME, padding="same")
            self.res_bn = BatchNormalization()

    def call(self, inputs):
        res = inputs

        x = self.conv_1(inputs)
        x = self.bn_1(x)
        x = tf.nn.relu(x)
        x = self.conv_2(x)
        x = self.bn_2(x)

        if self.__down_sample:
            res = self.res_conv(res)
            res = self.res_bn(res)

        # if not perform down sample, then add a shortcut directly
        x = self.merge([x, res])
        out = tf.nn.relu(x)
        return out


class ResNet18_LSTM(Model):

    def __init__(self, inputs, num_classes, **kwargs):
        """
            num_classes: number of classes in specific classification task.
        """
        super().__init__(**kwargs)
        self.conv_1 = Conv2D(64, (5, 5), strides=1,
                             padding="same", kernel_initializer="he_normal")
        self.init_bn = BatchNormalization()
        self.pool_2 = MaxPool2D(pool_size=(2, 2), strides=1, padding="same")
        self.res_1_1 = ResnetBlock(64)
        self.res_1_2 = ResnetBlock(64)
        self.res_2_1 = ResnetBlock(128, down_sample=True)
        self.res_2_2 = ResnetBlock(128)
        self.res_3_1 = ResnetBlock(256, down_sample=True)
        self.res_3_2 = ResnetBlock(256)
        self.res_4_1 = ResnetBlock(512, down_sample=True)
        self.res_4_2 = ResnetBlock(512)
        self.avg_pool = GlobalAveragePooling2D()
        self.flat = Flatten()
        self.fc = Dense(num_classes, activation="softmax")
        self.fc2 = Dense(128, activation = 'relu')
        self.lstm = LSTM(64, return_sequences=True)
        self.att_pool = SelfAttentivePooling()

    def call(self, inputs):
        out = self.conv_1(inputs)
        out = self.init_bn(out)
        out = tf.nn.relu(out)
        out = self.pool_2(out)
        for res_block in [self.res_1_1, self.res_1_2, self.res_2_1, self.res_2_2, self.res_3_1, self.res_3_2, self.res_4_1, self.res_4_2]:
            out = res_block(out)
        # out = self.avg_pool(out)
        # out = self.flat(out)
        # out = self.fc(out)
        x = TimeDistributed(Flatten())(out)
        x = self.lstm(x)
        x = self.att_pool(x)
        # x = self.fc2(x)
        # x = BatchNormalization()(x)
        output = self.fc(x)
        return output


def build_resnet_lstm(shape: List[int], n_label: int = 2) -> tf.keras.Model:
  return ResNet18_LSTM(inputs=input, num_classes=n_label)