# Freewire: Freely Wired Neural Networks
This repository is a collection of code for defining, visualizing, and training
freely wired neural networks. Freely wired neural networks operate at the level of individual nodes (or "neurons") and connections between them, instead of at
the level of homogeneous layers.

The SOTA in deep learning is generally propelled by unique neural network architectures. For 
example, residual connections as implemented by [ResNet](https://arxiv.org/abs/1512.03385) 
and [DenseNet](https://arxiv.org/abs/1608.06993) make it feasible to train larger neural networks
and achieve greater accuracy than models with layers that only operate on the representation
of the previous layer.

The recent work from [Facebook](https://arxiv.org/abs/1904.01569) shows that some randomly
wired neural networks have competitive performance on ImageNet. This repository
is a starting point for exploring how to design and optimize neural networks
that can be wired in very novel ways at the level of individual artificial neurons.

### Parallel Operations, not Layers
Instead of viewing a network as a series of layers that each have their own representation,
freely wired neural networks carry out a series of parallelized operations that extend a
1D tape of numbers. This is a generalization of what layers already do (if their representations were flattened), 
but doesn't specify that the inputs and activation functions of each node in a given operation need to be homogeneous. 
This graphic shows the 1D tape on the left and the freely wired
neural network that it represents on the right (biases are left out in this image for simplicity).

<img src="https://i.imgur.com/ouGgwEQ.png" height="300"><img src="https://i.imgur.com/13KNQ6f.png" height="300">

The goal of Freewire is to make it so that any arbitrary DAG of artificial neurons 
can be defined first and the optimized set of operations can be generated at runtime
and run on CUDA.


### Example -- XOR Gate
```python
from graph import Node, Graph
from model import Network

# node with no arguments is an input node
inputs = [Node(), Node()]
# pass a list of input nodes as the first argument
hidden = [Node(inputs, activation='sigmoid') for _ in range(0, 5)]
output = Node(hidden, activation='sigmoid')
# specify which nodes are inputs, hidden, or output nodes when generating graph
g = Graph(inputs, hidden, [output])
net = Network(g)
data = [
  [0, 0],
  [1, 0],
  [0, 1],
  [1, 1]
]
target = [0, 1, 1, 0]

net.compile('sgd', 'mse')
net.fit(data, target, epochs=10000, batch_size=1)
print("0 xor 0:", net([0, 0]))
print("0 xor 1:", net([0, 1]))
print("1 xor 0:", net([1, 0]))
print("1 xor 1:", net([1, 1]))
```
### Examples
See the `examples` folder for demos with the MNIST and CIFAR-10 datasets. (CIFAR-10 not implemented yet)

### Installation
`git clone` this repository and install project requirements with `pip install -r requirements.txt`
Requirements are:
* numpy
* torch==1.2.0
* graphviz
* pydot

Then install this package with `pip install .`

### Development
I am still actively developing the code here. See [todo.md](todo.md).