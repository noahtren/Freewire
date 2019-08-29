from freewire.graph import Node, Graph
from freewire.model import Network

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
