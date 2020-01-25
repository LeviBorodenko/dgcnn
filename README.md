# DGCNN [TensorFlow]
TensorFlow 2 implementation of _An end-to-end deep learning architecture for graph classification_ based on work by [M. Zhang et al., 2018](https://www.cse.wustl.edu/~muhan/papers/AAAI_2018_DGCNN.pdf).

Moreover, we offer an attention based modification of the above by utilising graph attention [(Veličković et al., 2017)](https://arxiv.org/abs/1710.10903) to learn edge weights.

### Installation

Simply run `pip install dgcnn`. The only dependency is `tensorflow>=2.0.0`.

### Usage

The core data structure is the _graph signal_. If we have N nodes in a graph each having C observed features then the graph signal is the tensor with shape (batch, N, C) corresponding to the data produced by all nodes. Often we have sequences of graph signals in a time series. We will call them _temporal_ graph signals and assume a shape of (batch, time steps, N, C).
For each graph signal we also need to have the corresponding adjacency matrices of shape (batch, N, N) or (batch, timesteps, N, N) for temporal and non-temporal data, respectively. While DGCNNs can operate on graphs with different node-counts, C should always be the same and each batch should only contain graphs with the same number of nodes.

#### The `DeepGraphConvolution` Layer

This adaptable layer contains the whole DGCNN architecture and operates on both temporal and non-temporal data. It takes the graph signals and their corresponding adjacency matrices and performs the following steps (as described in the paper):

We initialize the layer by providing  <a href="https://www.codecogs.com/eqnedit.php?latex=k,&space;c_1,&space;\dots,&space;c_h&space;\in&space;\mathbb{N}" target="_blank"><img style="vertical-align: middle" src="https://latex.codecogs.com/gif.latex?k,&space;c_1,&space;\dots,&space;c_h&space;\in&space;\mathbb{N}" title="k, c_1, \dots, c_h \in \mathbb{N}" /></a>. The layer has many optional parameters that are described in the table below.

1. It iteratively applies `GraphConvolution` layers h times with variable hidden feature dimensions <a href="https://www.codecogs.com/eqnedit.php?latex=c_i" target="_blank"><img style="vertical-align: middle" src="https://latex.codecogs.com/gif.latex?c_i" title="c_i" /></a>.

2. After that, it concatenates all the outputs of the graph convolutions into one tensor which has the shape (..., N, <a href="https://www.codecogs.com/eqnedit.php?latex=\sum_{i&space;=&space;1}^hc_i" target="_blank"><img style="vertical-align: middle" src="https://latex.codecogs.com/gif.latex?\sum_{i&space;=&space;1}^hc_i" title="\sum_{i = 1}^hc_i" /></a>).

3. Finally it applies `SortPooling` as described in the paper to obtain the output tensor of shape (..., k, <a href="https://www.codecogs.com/eqnedit.php?latex=\sum_{i&space;=&space;1}^hc_i" target="_blank"><img style="vertical-align: middle" src="https://latex.codecogs.com/gif.latex?\sum_{i&space;=&space;1}^hc_i" title="\sum_{i = 1}^hc_i" /></a>).

Import this layer with `from gdcnn.components import DeepGraphConvolution`.

Initiated it with the following parameters:

| Parameter | Function |
|:------------- | :--------|
|`hidden_conv_units` (required) | List of the hidden feature dimensions used in the graph convolutions. <a href="https://www.codecogs.com/eqnedit.php?latex=k,&space;c_1,&space;\dots,&space;c_h&space;\in&space;\mathbb{N}" target="_blank"><img style="vertical-align: middle" src="https://latex.codecogs.com/gif.latex?c_1,&space;\dots,&space;c_h" title="c_1, \dots, c_h" /></a> in the paper.|
|`k` (required) |Number of nodes to be kept after SortPooling.|
|`flatten_signals` (default: False) | If `True`, flattens the last 2 dimensions of the output tensor into 1|
|`attention_heads` (default: None) | If given, then instead of using <a href="https://www.codecogs.com/eqnedit.php?latex=D^{-1}E" target="_blank"><img src="https://latex.codecogs.com/gif.latex?D^{-1}E" title="D^{-1}E" /></a> as the transition matrix inside the graph convolutions, we will use an attention based transition matrix. Utilizing `dgcnn.attention.AttentionMechanism` as the internal attention mechanism. This sets the number of attention heads used.|
|`attention_units` (default: None) | Also needs to be provided if `attention_heads` is set. This is the size of the internal embedding used by the attention mechanism.|

Thus, if we have non-temporal graph signals with 10 nodes and 5 features each and we would like to apply a DGCNN containing 3 graph convolutions with hidden feature dimensions of 10, 5 and 2 and SortPooling that keeps the 5 most relevant nodes. Then we would run

```python
from dgcnn.components import DeepGraphConvolution
from tensorflow.keras.layers import Input
from tensorflow.keras import Model


# generating random graph signals as test data
graph_signal = np.random.normal(size=(100, 10, 5)

# corresponding fully connected adjacency matrices
adjacency = np.ones((100, 10, 10))

# inputs to the DGCNN
X = Input(shape=(10, 5), name="graph_signal")
E = Input(shape=(10, 10), name="adjacency")

# DGCNN
# Note that we pass the signals and adjacencies as a tuple.
# The graph signal always goes first!
output = DeepGraphConvolution([10, 5, 2], k=5 )((X, E))

# defining model
model = Model(inputs=[X, E], outputs=output)
```

#### Further layers and features

The documentation contains information on how to use the internal `SortPooling`, `GraphConvolution` and `AttentionMechanism` layers and also describes more optional parameters like regularisers, initialisers and constrains that can be used.

### Contribute
Bug reports, fixes and additional features are always welcome! Make sure to run the tests with `python setup.py test` and write your own for new features. Thanks.
