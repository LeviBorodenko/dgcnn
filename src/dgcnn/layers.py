"""Provides Graph Convolution and SortPooling Layers.

Based on https://www.cse.wustl.edu/~muhan/papers/AAAI_2018_DGCNN.pdf
"""
import tensorflow as tf
import tensorflow.keras.layers as layers
from tensorflow.keras import initializers, regularizers
from tensorflow.keras import constraints, activations

__author__ = "Levi Borodenko"
__copyright__ = "Levi Borodenko"
__license__ = "mit"


class GraphConvolution(layers.Layer):
    """docstring for GraphConvolution"""

    def __init__(self,
                 num_hidden_features: int = 1,
                 activation: str = "tanh",
                 weights_initializer: str = "GlorotNormal",
                 weights_regularizer=None,
                 weights_constraint=None,
                 dropout_rate: float = None
                 ):
        super(GraphConvolution, self).__init__()

        # make sure the number of hidden features is > 0
        if num_hidden_features < 1:
            raise ValueError("num_hidden_features must be a positive integer.")

        # store number of hidden features. "c'" in paper.
        self.c_dash = int(num_hidden_features)

        # get regularizer, initializer and constraint for weight matrix W
        self.weights_initializer = initializers.get(weights_initializer)
        self.weights_regularizer = regularizers.get(weights_regularizer)
        self.weights_constraint = constraints.get(weights_constraint)

        # get activation. f in paper.
        self.f = activations.get(activation)

        # check whether dropout should be used
        if dropout_rate is None:
            self.use_dropout = False
        else:

            # Raise error if dropout rate not in [0,1]
            if not (0 <= dropout_rate <= 1):
                raise ValueError("Dropout-rate should be in [0,1]")

    def build(self, input_shape):

        # Expect input to be a tuple containing
        # (X, T) where X is the (n x c) graph signal and
        # T is the (n x n) transition matrix.
        # T is D^-1*A in the paper.
        X_shape, T_shape = input_shape

        # As the number of nodes is varying, we only care
        # about the number of features. c in paper.
        self.c = X_shape[-1]

        # build (c x c') weight matrix W
        self.W = self.add_weight(
            shape=(self.c, self.c_dash),
            initializer=self.weights_initializer,
            regularizer=self.weights_regularizer,
            constraint=self.weights_constraint,
            name="graph_conv_weight_matrix",
            trainable=True,
        )
