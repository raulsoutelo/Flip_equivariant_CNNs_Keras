import tensorflow as tf
from tensorflow.keras.layers import Conv2D, InputSpec
from tensorflow.keras import backend as K
from tensorflow.python.ops import nn
import functools
import six

from utils import is_equivariant, horizontal_flip, transformation_feature_space

class Conv2D_HF2(Conv2D):
    """
    In this class I have taken the Conv2D implementation from
    https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/keras/layers/convolutional.py
    and modified the methods build and call to have a 2D convolutional layer equivariant to transformation
    in input space 'g'. 
    'g' is the transformation in output space when applying horizontal flip in input space.
    """

    def build(self, input_shape):
        input_shape = tf.TensorShape(input_shape)
        input_channel = self._get_input_channel(input_shape)
        kernel_shape = self.kernel_size + (input_channel // 2,
                                       self.filters)

        self.kernel = self.add_weight(
            name='kernel',
            shape=kernel_shape,
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
            trainable=True,
            dtype=self.dtype)
        if self.use_bias:
            self.bias = self.add_weight(
                name='bias',
                shape=(self.filters,),
                initializer=self.bias_initializer,
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint,
                trainable=True,
                dtype=self.dtype)
        else:
            self.bias = None
        channel_axis = self._get_channel_axis()
        self.input_spec = InputSpec(min_ndim=self.rank + 2,
                                    axes={channel_axis: input_channel})

        # Convert Keras formats to TF native formats.
        if self.padding == 'causal':
            tf_padding = 'VALID'  # Causal padding handled in `call`.
        elif isinstance(self.padding, six.string_types):
            tf_padding = self.padding.upper()
        else:
            tf_padding = self.padding
        tf_dilations = list(self.dilation_rate)
        tf_strides = list(self.strides)

        tf_op_name = self.__class__.__name__
        if tf_op_name == 'Conv1D':
            tf_op_name = 'conv1d'  # Backwards compat.

        self._convolution_op = functools.partial(
            tf.nn.convolution,
            strides=tf_strides,
            padding=tf_padding,
            dilations=tf_dilations,
            data_format=self._tf_data_format,
            name=tf_op_name)
        self.built = True


    def call(self, inputs):

        inputs_first = inputs[:,:,:, :inputs.shape[3]//2]
        inputs_second = inputs[:,:,:, inputs.shape[3]//2:]

        outputs1 = self._convolution_op(inputs_first, self.kernel) 
        outputs2 = self._convolution_op(inputs_second, K.reverse(self.kernel,axes=1)) 
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
    conv2 = Conv2D_HF2(filters=num_filters,
                    kernel_size=kernel_size,
                    bias_initializer="random_uniform")

    is_equivariant(f=conv2, g_0=transformation_feature_space, g_1=transformation_feature_space)