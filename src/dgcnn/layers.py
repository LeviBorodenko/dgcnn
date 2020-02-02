"""Provides Graph Convolution and SortPooling Layers.

Based on https://www.cse.wustl.edu/~muhan/papers/AAAI_2018_DGCNN.pdf
"""
import tensorflow as tf
import tensorflow.keras.layers as layers
from tensorflow.keras import activations, constraints, initializers, regularizers

__author__ = "Levi Borodenko"
__copyright__ = "Levi Borodenko"
__license__ = "mit"


class GraphConvolution(layers.Layer):
    """Graph convolution layer as described by M. Zhang et al., 2018.

    A slight difference is that, instead of taking a graph signal and a
    adjacency matrix, this layer takes a graph signal and a transition matrix
    on the underlying graph.
    In the paper the transition matrix used was simply D^-1*E.

    Inputs:
        tuple (X, T) where:

        X (tensor): Tensor of shape (batch, timesteps (optional), N, F).
        These are the (temporal) graph signals.

        T (tensor): Shape (batch, timesteps (optional), N, N).
        The corresponding transition matrices.

    Keyword Arguments:
        num_hidden_features (int):
            - c' in paper.
            - Number of features in the weight matrix W.
            - (default: 1)

        activation (str): f in paper. Should be injective! (default: "tanh")

        weights_initializer (str): initializer for W (default:"GlorotNormal")

        weights_regularizer ([type]): regularizer for W (default: None)

        weights_constraint ([type]): constraint on W (default: None)

        dropout_rate (float): drop-out rate to be used on W. (default: None)

    Extends:
        tf.keras.layers.Layer

    References:
        Zhang, M., Cui, Z., Neumann, M. and Chen, Y., 2018, April.
        An end-to-end deep learning architecture for graph classification.
        In Thirty-Second AAAI Conference on Artificial Intelligence.

        https://www.cse.wustl.edu/~muhan/papers/AAAI_2018_DGCNN.pdf
    """

    def __init__(
        self,
        num_hidden_features: int = 1,
        activation: str = "tanh",
        weights_initializer: str = "GlorotNormal",
        weights_regularizer=None,
        weights_constraint=None,
        dropout_rate: float = None,
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

    def call(self, inputs):

        # Expect tuple (Graph signal, Transition Matrix) = (X, T)
        X, T = inputs

        # as in the paper, we calculate the output Z as
        # Z = f(T * X * W)
        Z = self.f(tf.matmul(T, tf.matmul(X, self.W)))

        return Z


class SortPooling(layers.Layer):
    """SortPooling layer returning the top k most relevant nodes.

    Inputs:
        tuple (Z, E) where:

            - Z (tensor): Tensor of shape (batch, timesteps (optional), N, F).
              The concatenated graph convolution outputs.

            - E (tensor): Shape (batch, timesteps (optional), N, N). The
              corresponding adjacency matrices.

    Arguments:
        k (int): k in paper. Number of nodes in the output signal.

    Raises:
        ValueError: If k is not a positive integer.

    Extends:
        tf.keras.layers.Layer

    References:
        Zhang, M., Cui, Z., Neumann, M. and Chen, Y., 2018, April.
        An end-to-end deep learning architecture for graph classification.
        In Thirty-Second AAAI Conference on Artificial Intelligence.

        https://www.cse.wustl.edu/~muhan/papers/AAAI_2018_DGCNN.pdf
    """

    def __init__(self, k: int):
        super(SortPooling, self).__init__()

        # Number of nodes to be kept (k in paper)
        k = int(k)
        if k <= 0:
            raise ValueError("K must be a positive integer")
        self.k = k

    def call(self, inputs):

        # Takes the concatenated & convolved signals Z
        Z = inputs

        # dimensionality of Z
        Z_dim = len(Z.shape)

        # get number of nodes
        n = tf.shape(Z)[-2]

        # Sort last column and return permutation of indices
        sort_perm = tf.argsort(Z[..., -1])

        # Gather rows according to the sorting permutation
        # thus sorting the rows according to the last column
        Z_sorted = tf.gather(Z, sort_perm, axis=-2, batch_dims=Z_dim - 2)

        # cast Z_sorted into float32
        Z_sorted = tf.cast(Z_sorted, tf.float32)

        def truncate():
            """If we have more nodes than we want to keep,
            then we simply truncate.
            """

            # trim number of nodes to k if k < n
            Z_out = Z_sorted[..., : self.k, :]

            return Z_out

        def pad():
            """If we have less nodes than we would like to keep,
            then we simply pad with empty nodes.
            """

            # if temporal signal:
            padding = [[0, 0], [0, self.k - n], [0, 0], [0, 0]]

            # if normale signal
            if Z_dim == 3:
                padding = [[0, 0], [0, self.k - n], [0, 0]]

            # padded output
            Z_out = tf.pad(Z_sorted, padding)

            return Z_out

        Z_out = tf.cond(tf.less_equal(self.k, n), truncate, pad)

        return Z_out
