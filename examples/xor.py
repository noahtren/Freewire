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

m.compile(optimizer='sgd', loss_function='mse')
m.fit(data, target, epochs=10000, batch_size=1)
print("0 xor 0:", m([0, 0]))
print("0 xor 1:", m([0, 1]))
print("1 xor 0:", m([1, 0]))
print("1 xor 1:", m([1, 1]))
