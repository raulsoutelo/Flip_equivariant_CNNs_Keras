import tensorflow as tf
from tensorflow.keras.layers import Conv2D
from tensorflow.keras import backend as K
from tensorflow.python.ops import nn

from utils import is_equivariant, horizontal_flip, transformation_feature_space

class Conv2D_HF(Conv2D):
    """
    In this class I have taken the Conv2D implementation from
    https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/keras/layers/convolutional.py
    and modified the method call to have a 2D convolutional layer equivariant to flip.
    """

    def call(self, inputs):

        outputs1 = self._convolution_op(inputs, self.kernel) 
        outputs2 = self._convolution_op(inputs, K.reverse(self.kernel,axes=1)) 
        outputs = K.concatenate([outputs1, outputs2], axis=3)
        if self.use_bias:
            outputs = nn.bias_add(
                            outputs, K.concatenate([self.bias, self.bias], axis=0), 
                            data_format=self._tf_data_format)

        return outputs

if __name__== "__main__":

    # create function to be checked if equivariant
    num_filters = 10
    kernel_size = 3
    conv2 = Conv2D_HF(filters=num_filters,
                    kernel_size=kernel_size,
                    bias_initializer="random_uniform")

    is_equivariant(f=conv2, g_0=horizontal_flip, g_1=transformation_feature_space)