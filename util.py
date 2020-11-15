import tensorflow as tf
import numpy as np

class SqueezeLayer(tf.keras.layers.Layer):
    """ a class that squeezes a given axis of a tensor"""

    def __init__(self):
        super(SqueezeLayer, self).__init__()

    def call(self, input_tensor, axis=3):
        try:
            output = tf.squeeze(input_tensor, axis)
        except:
            output = input_tensor
        return output


class DotLayer(tf.keras.layers.Layer):
    """ Return cosine similarity between two columns of two matrices. """

    def __init__(self):
        super(DotLayer, self).__init__()

    def call(self, list_tensors):
        layer = tf.keras.layers.Dot(axes=[2, 2], normalize=True)
        output_dot = layer(list_tensors)
        output_diag = tf.matrix_diag_part(output_dot)
        return output_diag


class DownsampleLayer(tf.keras.layers.Layer):
    """ a class that downsamples a given axis of a tensor"""

    def __init__(self):
        super(DownsampleLayer, self).__init__()

    def call(self, input_tensor, rate=3):
        output = input_tensor[:,0::rate,:,:]

        return output


# our defined loss function based on binary cross entropy
def loss_BCE_custom():
    """
    Return binary cross entropy loss for cosine similarity layer.

    :param cos_scores_sig: array of float numbers, output of the cosine similarity
        layer followed by sigmoid function.
    :return: a function, which will be used as a loss function in model.compile.
    """

    def loss(y_true, y_pred):
        print(tf.keras.backend.int_shape(y_pred))
        part_pos = tf.keras.backend.sum(-y_true * tf.keras.backend.log(y_pred), axis= -1)
        part_neg = tf.keras.backend.sum((y_true-1)*tf.keras.backend.log(1-y_pred), axis= -1)
        return (part_pos + part_neg) / tf.keras.backend.int_shape(y_pred)[-1]

    return loss