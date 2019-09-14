# Freewire: Freely Wired Neural Networks
Freewire is a Keras-like API for creating optimized freely wired neural networks to run
on CUDA. Freely wired neural networks are defined at the level of individual nodes (or "neurons") 
and their connections, instead of at the level of homogeneous layers.
The goal of Freewire is to make it so that any arbitrary DAG of artificial neurons 
can be defined first and the optimized set of operations can be generated at runtime
and run on CUDA.

This repository is a starting point for exploring how to design and optimize neural networks
that can be wired in very novel ways at the level of individual artificial neurons, while
retaining the speed and memory efficiency of traditional neural networks.

### Parallel Operations, not Layers
Instead of viewing a network as a series of layers that each have their own representation,
freely wired neural networks carry out a series of parallelized operations that extend a
flat, 1D tape of numbers. This is a generalization of what layers already do (if their representations were flattened), 
but doesn't specify that the inputs and activation functions of each node in a given operation need to be homogeneous. 
This graphic shows the 1D tape on the left and the freely wired
neural network that it represents on the right (biases are left out in this image for simplicity).
Also note than the 1D tape is extended to 2D to allow training in batches.

<img src="https://i.imgur.com/ouGgwEQ.png" height="300"><img src="https://i.imgur.com/13KNQ6f.png" height="300">

### XOR Gate Example
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

```
git clone https://github.com/noahtren/Freewire
cd Freewire
pip install -e .
```

This will automatically install the requirements in `requirements.txt`:
* numpy
* torch==1.2.0
* graphviz
* pydot
