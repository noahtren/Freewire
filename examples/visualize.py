from freewire import Node, Graph, Model
from freewire import visualize
inputs = [Node(), Node()]
hidden1 = Node(inputs)
hidden2 = Node([inputs[0], hidden1])
hidden3 = Node([inputs[1], hidden1])
output = Node([hidden2, hidden3])
g = Graph(inputs, [hidden1, hidden2, hidden3], [output])
visualize(g, title="architecture")
m = Model(g, initialization="he")
visualize(g, title="architecture_and_weights")