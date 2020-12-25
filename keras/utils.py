import tensorflow as tf
from tensorflow.keras import backend as K

def horizontal_flip(x):
    """
    This function flips an image horizontally. 
    The third dimension corresponds to columns.
    """
    return K.reverse(x,axes=2) 
    
def transformation_feature_space(x):
    """
    This function performs a transformation in feature space:
    The transformation does the following:
    1) Switches first and second half of the embedding 
    in the channels output dimension
    2) Flips the the embedding in third axis (columns)
    """
    x_first = x[:,:,:, :x.shape[3]//2]
    x_second = x[:,:,:, x.shape[3]//2:]
    x = K.concatenate([x_second, x_first], axis=3)
    x = K.reverse(x,axes=2)
    return x

def is_equivariant(f, g_0, g_1):
    """
    This function checks if a function 'f' is equivariant
    and prints the result.
    A function f is equivariant to transformation g_0 if
    there exist g_1 such that f(g_0(x)) = g_1(f(x))
    Args:
        f: function to be checked if it is equivariant
        g_0: transformation in input space
        g_1: transformation in output space
    """

    batch = 15
    channels = 12
    rows = 10
    cols = 10

    # Input and output dimensions of the convolutional layer (function f):
    #input (batch, rows, cols, channels)
    #output (batch, new_rows, new_cols, filters)

    # original input
    input1 = tf.random.uniform(shape=(batch, rows, cols, channels))

    # transformed input
    input2 = g_0(input1)

    # apply function f
    output1 = f(input1)
    output2 = f(input2)

    # transformed output
    transformed_output1 = g_1(output1)

    # check if the two embeddings are equal
    diff_abs = tf.abs(output2 - transformed_output1)
    result = tf.reduce_all(diff_abs < 1e-6).numpy()

    print('is the function f equivariant? ' + str(result))