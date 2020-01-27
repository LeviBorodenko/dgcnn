"""Implementation of Deep Graph Convolution with SortPooling.

Based on https://www.cse.wustl.edu/~muhan/papers/AAAI_2018_DGCNN.pdf
"""
import tensorflow as tf
import tensorflow.keras.layers as layers

from dgcnn.attention import AttentionMechanism
from dgcnn.layers import GraphConvolution, SortPooling
from dgcnn.utils import is_positive_integer

__author__ = "Levi Borodenko"
__copyright__ = "Levi Borodenko"
__license__ = "mit"


class DeepGraphConvolution(layers.Layer):
    """Layer that performs deep graph convolution and
    SortPooling as described by M. Zhang et al., 2018.

    Arguments:

        hidden_conv_units (list):

            - c_1, ..., c_h in paper.

            - Hidden feature dimensions of the recursively
            applied graph convolutions.

        k (int):

            - k in paper.

            - Number of nodes to be kept after SortPooling.

        **kwargs:

            - arguments to be passed to the GraphConvolution layers inside.

    Keyword Arguments:

        flatten_signals (bool):
            - Flattens the last 2 dimensions of the output
              tensor into 1. So that is reshaped from (..., k, sum(c_i)) to
              (..., k * sum(c_i)).

            - (Default: False)

        attention_heads (int):

            - If given, then instead of using D^-1 E as the
              transition matrix inside the graph convolutions, we will use
              an attention based transition matrix. We use
              dgcnn.attention.AttentionMechanism as the internal attention
              mechanism.

            - Sets the number of attention heads to be used.

            - (default: (None))

        attention_units (int):

            - If given, then instead of using D^-1 E as the
            transition matrix inside the graph convolutions, we will use
            an attention based transition matrix.
            We use dgcnn.attention.AttentionMechanism as the internal attention
            mechanism.

            - This sets the size of the hidden representation used by
            the attention mechanism.

            - (default: (None)).

        use_sortpooling (bool):

            - If False, won't apply SortPooling at the end of the procedure.

            - (default: True)

    Inputs:

        - X (tf.Tensor):

            - The (temporal) graph signals.

            - Should have shape (batch, N, F) for graph signals with
              N nodes and F features or (batch, timesteps, N, F) for
              temporal graph signals.

        - E (tf.Tensor):

            - Corresponding adjacency matrix.

            - Should have shape (batch, N, N) or (batch, timesteps, N, N).

    Returns:

        tf.Tensor: Z in paper. Shape (batch, (timesteps), k, sum c_i).
        This is the transformed graph signal that we obtain by concatenating
        the outputs of the recursive convolutions and applying SortPooling.

    Example::

        # generating random temporal graph signals
        graph_signal = np.random.normal(size=(100, 5, 10, 5)

        # corresponding fully connected adjacency matrices
        adjacency = np.ones((100, 5, 10, 10))

        # corresponding labels
        labels = np.ones((100, 5, 5, 10))

        # creating tensorflow dataset
        dataset = tf.data.Dataset.from_tensor_slices(
                    (
                        {
                            "graph_signal": graph_signal,
                            "adjacency": adjacency
                        },
                        labels
                        )
                    ).batch(2)

        # defining inputs
        X = Input(shape=(5, 10, 5), name="graph_signal")
        E = Input(shape=(5, 10, 10), name="adjacency")

        # DGCNN
        output = DeepGraphConvolution(
            [5, 3, 2],
            k=5,
            )((X, E))

        # defining model
        model = Model(inputs=[X, E], outputs=output)


    References:

        Zhang, M., Cui, Z., Neumann, M. and Chen, Y., 2018, April.
        An end-to-end deep learning architecture for graph classification.
        In Thirty-Second AAAI Conference on Artificial Intelligence.

        https://www.cse.wustl.edu/~muhan/papers/AAAI_2018_DGCNN.pdf

    Extends:

        tf.keras.layers.Layer
    """

    def __init__(
        self,
        hidden_conv_units: list,
        k: int,
        attention_heads: int = None,
        attention_units: int = None,
        flatten_signals: bool = False,
        use_sortpooling: bool = True,
        **kwargs
    ):
        super(DeepGraphConvolution, self).__init__()

        # assert all quantities are of right type and range
        is_positive_integer(k, "Number of Nodes to keep (k)")

        try:
            for c in hidden_conv_units:
                is_positive_integer(c, "Number of hidden features (c_i)")
        except TypeError:
            raise ValueError("hidden_conv_units must be iterable of integers.")

        # check if we are using attention
        if attention_heads is None or attention_units is None:
            self.use_attention = False
        else:
            is_positive_integer(attention_heads, "attention_heads")
            is_positive_integer(attention_units, "attention_units")
            self.use_attention = True

        # store all as attributes
        self.k = k
        self.hidden_conv_units = hidden_conv_units
        self.attention_heads = attention_heads
        self.attention_units = attention_units
        self.flatten_signals = flatten_signals
        self.use_sortpooling = use_sortpooling

        # save kwargs to pass them to the graphconv layers
        self.kwargs = kwargs

    def build(self, input_shape):

        # to be populated by GraphConvolution layers
        self.convolutions = []

        # store input shape
        self.shape = input_shape

        for c in self.hidden_conv_units:
            layer = GraphConvolution(num_hidden_features=c, **self.kwargs)
            self.convolutions.append(layer)

        # initiating SortPooling layer
        if self.use_sortpooling:
            self.SortPooling = SortPooling(self.k)

        # create AttentionMechanism if required
        if self.use_attention:

            # initiate attention
            self.AttentionMechanism = AttentionMechanism(
                F=self.attention_units, num_heads=self.attention_heads
            )

    def call(self, inputs):

        # get graph signal and adjacency_matrix
        X, E = inputs

        # normalise rows of E if attention is not needed
        if not self.use_attention:

            # get degree matrix
            D = tf.reduce_sum(E, axis=-1, keepdims=True)

            # normalise E to get the natural transition matrix
            T = (1 / D) * E

        else:

            # use attention to generate transition matrix
            T = self.AttentionMechanism((X, E))

        # Now we have the transition matrix and the signal, we can apply
        # all the layers in sequence and concat their outputs
        rec_conv_signals = []
        for conv in self.convolutions:

            # recursively convolve the graph signal
            X = conv((X, T))
            rec_conv_signals.append(X)

        # concat them to (N x sum(c_i)) signal
        Z = tf.concat(rec_conv_signals, axis=-1)

        # if we do not sortpool or flatten then simply
        # return Z
        if not self.use_sortpooling and not self.flatten_signals:
            return Z

        # if we do not sortpool but want to flatten
        # then simply flatten Z and return it
        if not self.use_sortpooling and self.flatten_signals:

            # Wanting to flatten means we convert our Z that has shape
            # (..., n, sum(c_i)) to (..., n * sum(c_i)).

            # shape that we need
            ####

            # (None) or (None, None) for non-temporal or
            # temporal data
            outer_shape = tf.shape(X)[:-2]
            inner_shape = tf.constant([X.shape[-2] * sum(self.hidden_conv_units)])

            shape = tf.concat([outer_shape, inner_shape], axis=0)

            Z = tf.reshape(Z, shape)

            return Z

        # apply SortPooling
        Z_pooled = self.SortPooling(Z)

        if not self.flatten_signals:

            # Creating goal shapes
            # outer_shape is (None) or (None, None)
            # for temporal or non temporal graph signals, resp.
            outer_shape = X.shape[:-2]

            # Inner shape the output of the SortPooling layer.
            # (k, sum(c_i)) in paper
            inner_shape = self.k, sum(self.hidden_conv_units)

            Z_pooled.set_shape(outer_shape + inner_shape)

            return Z_pooled
        else:

            # The aim here is to convert our output that has shape
            # (..., k, sum(c_i)) to (..., k * sum(c_i)) as is done
            # in the paper.
            # The difficulty is that ... contains None so to reshape
            # we need the actual shape as the data comes in and not
            # the inferred shape information that is in the comp. graph.
            # tf.shape allows us to access the true shape live.

            # shape that we need
            outer_shape = tf.shape(X)[:-2]
            inner_shape = tf.constant([self.k * sum(self.hidden_conv_units)])

            shape = tf.concat([outer_shape, inner_shape], axis=0)

            Z_pooled = tf.reshape(Z_pooled, shape)

            return Z_pooled
