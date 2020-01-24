import tensorflow as tf
from tensorflow.keras import initializers, regularizers
from tensorflow.keras.layers import Average, Layer

__author__ = "Levi Borodenko"
__copyright__ = "Levi Borodenko"
__license__ = "mit"


class GraphAttentionHead(Layer):
    """Returns an attention matrix based on the graph signal
    and the adjacency matrix.

    Corresponds to one attention head.

    References:
        https://arxiv.org/pdf/1710.10903.pdf

    Inputs:
        tuple (X, E) where:

        - X (tensor): Tensor of shape (batch, timesteps (optional), N, F).
          These are the (temporal) graph signals.

        - E (tensor): (batch, timesteps (optional), N, N). The corresponding
          adjacency matrices.

    Returns:
        tf.tensor: Tensor of same shape as E. Transition matrix on the graph.

    Arguments:
        F (int): Dimension of internal embedding. F' in paper.

    Keyword Arguments:
        kernel_initializer (str): (default: {"glorot_uniform"})
        attn_vector_initializer (str): (default: {"glorot_uniform"})
        kernel_regularizer: (default: {None})
        attn_vector_regularizer: (default: {None})

    Extends:
        tf.keras.layers.Layer

    """

    def __init__(
        self,
        F: int,
        kernel_initializer="glorot_uniform",
        attn_vector_initializer="glorot_uniform",
        kernel_regularizer=None,
        attn_vector_regularizer=None,
        **kwargs
    ):
        super(GraphAttentionHead, self).__init__()

        # Number of features we extract and then
        # recombine to generate the attention.
        # (F` in paper)
        self.F = F

        # storing initializers
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.attn_vector_initializer = initializers.get(attn_vector_initializer)

        # storing regularizes
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.attn_vector_regularizer = regularizers.get(attn_vector_regularizer)

    def build(self, input_shape):
        # we expect the input to be (Graph signal, Adjacency matrix)
        # Extracting dimensions of graph signal
        # Number of features per node
        self.K = input_shape[0][-1]

        # Check if we have a time series of graph signals
        if len(input_shape[0]) > 3:
            self.is_timeseries = True
        else:
            self.is_timeseries = False

        # initializing kernel
        # W in paper
        self.W = self.add_weight(
            shape=(self.K, self.F),
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            name="attn_kernel",
            trainable=True,
        )

        # in the paper we need to calculate
        # [X_i*W || X_j*W] v were v is 2F dimensional
        # we skip the concatenation by decomposing v into
        # v1, v2 in R^F and thus writing the above as
        # X_i*W*v1 + X_j*W*v2
        self.v_1 = self.add_weight(
            shape=(self.F, 1),
            initializer=self.attn_vector_initializer,
            regularizer=self.attn_vector_regularizer,
            name="attn_vector_1",
            trainable=True,
        )
        self.v_2 = self.add_weight(
            shape=(self.F, 1),
            initializer=self.attn_vector_initializer,
            regularizer=self.attn_vector_regularizer,
            name="attn_vector_2",
            trainable=True,
        )

    def call(self, inputs):

        # get graph signal corresponding adjacency matrix
        X, E = inputs

        # calculate attentive transition matrix
        ###

        # If X is the graph signal then note that
        # doing the following is equivalent to the matrix in eq (1)
        # :

        # 1. calculate X*W where X is the (N x K) graph signal
        # and W is the (K x F) kernel matrix
        # 2. let v1 and v2 be two (F x 1) vectors and find
        # d1 = X*W*v1, d2 = X*W*v2 (N x 1)
        # 3. Using numpys broadcasting rules we now calculate
        # A = d1 + d2^T which will be (N x N)

        # 1.
        # Affine project each feature from R^K to R^F
        # using the kernel (W in paper)
        proj_X = tf.matmul(X, self.W)

        # 2.
        # multiply with v1 and v2
        d1 = tf.matmul(proj_X, self.v_1)  # (N x 1)

        d2 = tf.matmul(proj_X, self.v_2)

        # 3.
        # create an (N x N) matrix of pairwise sums of entries from
        # d1 and d2.
        # We utilise numpy broadcasting to achieve that
        # Note: we need to specify that we only transpose
        # the last 2 dimensions of d2 which due to batch-wise
        # data can have 3 dimensions.
        if self.is_timeseries:
            A = d1 + tf.transpose(d2, perm=[0, 1, 3, 2])
        else:
            A = d1 + tf.transpose(d2, perm=[0, 2, 1])

        # The above A is the unnormalized attention matrix.
        # first we remove all entries in A that correspond to edges that
        # are not in the graph.
        A = tf.multiply(A, E)

        # apply non linearity (as in paper: LeakyReLU with a=0.2)
        A = tf.nn.leaky_relu(A, alpha=0.2)

        # now we softmax this matrix over its columns to normalise it.
        A = tf.nn.softmax(A)

        return A


class AttentionMechanism(Layer):
    """Attention Mechanism.

    We use multiple attention heads and average their outputs.
    Takes a graph signal and its adjacency matrix and returns
    an attentive transition matrix.

    Arguments:
        F (int): Dimensions of hidden representation used for attention.

    Keyword Arguments:
        num_heads (int): Number of attention heads to be used. (default: {1})

    Inputs:
        tuple containing (X, E):

            - X (tensor):  (batch, timesteps, N, F), graph signals on N nodes
              with F features. Also works with individual graph signals with no
              time steps, i.e. (batch, N, F).

            - E (tensor):  (batch, timesteps, N, N), corresponding
              adjacency matrices.

    Returns:
        tf.tensor: Tensor of shape (batch, timesteps, N, N), corresponding to
        attentive transition matrices.

    Example::

        # creating train data
        x_train = np.random.normal(size=(1000, 10, 10, 2))
        y_train = np.ones((1000, 10, 2, 10, 10))

        # corresponding random adjacency_matrix
        E = np.random.randint(0, 2, size=(1000, 2, 10, 10))

        # building tiny model
        X = layers.Input(shape=(None, 10, 2))
        A = AttentionMechanism(F=5,
                               num_heads=5)((X, E))

        model = keras.Model(inputs=X, outputs=A)

    Extends:
        tf.keras.layers.Layer
    """

    def __init__(self, F: int, num_heads: int = 1, **kwargs):
        super(AttentionMechanism, self).__init__()

        # Number of hidden units for Attention Mechanism
        self.F = F

        # number of attention heads
        self.num_heads = num_heads

        # populated by GraphAttentionHead layers
        self.attn_heads = []

        for _ in range(num_heads):

            # get Graph Attention Head layer
            attn_head = GraphAttentionHead(F=F, **kwargs)

            self.attn_heads.append(attn_head)

    def call(self, inputs):

        attention_layers = []

        # apply all attention layers to inputs
        for layer in self.attn_heads:

            attention_layers.append(layer(inputs))

        # now average all their outputs
        if self.num_heads > 1:
            return Average()(attention_layers)
        else:
            return attention_layers[0]
