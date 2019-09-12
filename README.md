# Freewire: Freely Wired Neural Networks
This repository is a collection of code for defining, visualizing, and training
optimized freely wired neural networks. Freely wired neural networks operate at 
the level of individual nodes (or "neurons") and connections between them, instead of at
the level of homogeneous layers.

The SOTA in deep learning is often propelled by unique neural network architectures. For 
example, residual connections as implemented by [ResNet](https://arxiv.org/abs/1512.03385) 
and [DenseNet](https://arxiv.org/abs/1608.06993) make it feasible to train larger neural networks
and achieve greater accuracy than models with layers that only operate on the representation
of the previous layer.

The recent work from [Facebook](https://arxiv.org/abs/1904.01569) shows that some randomly
wired neural networks have competitive performance on ImageNet. Also, Google introduced
[Weight Agnostic Neural Networks](https://arxiv.org/abs/1906.04358) which focuses on evolving
neural network architectures with uniform weights between all connections.

This repository is a starting point for exploring how to design and optimize neural networks
that can be wired in very novel ways at the level of individual artificial neurons, while
retaining the ability to train them with backpropagation.

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
from freewire import Node, Graph
from freewire import Model

# node with no arguments is an input node
inputs = [Node(), Node()]
# pass a list of input nodes as the first argument
hidden = [Node(inputs, activation='sigmoid') for _ in range(0, 5)]
output = Node(hidden, activation='sigmoid')
# specify which nodes are inputs, hidden, or output nodes when generating graph
g = Graph(inputs, hidden, [output])
m = Model(g)
data = [
  [0, 0],
  [1, 0],
  [0, 1],
  [1, 1]
]
target = [0, 1, 1, 0]

m.compile('sgd', 'mse')
m.fit(data, target, epochs=10000, batch_size=1)
print("0 xor 0:", m([0, 0]))
print("0 xor 1:", m([0, 1]))
print("1 xor 0:", m([1, 0]))
print("1 xor 1:", m([1, 1]))
```
### More Examples
See the `examples` folder for more examples, including a network for MNIST with randomly wired layers.

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