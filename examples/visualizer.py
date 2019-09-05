from freewire import Node, Graph
from freewire import Model
from freewire import visualize

inputs = [Node() for _ in range(2)]
hidden = [Node(inputs) for _ in range(4)]
outputs = [Node(hidden)]

g = Graph(inputs, hidden, outputs)
visualize(g, title='Empty')

m = Model(g, initialization='he')
g = m.update_graph()
visualize(g, title='He')

m = Model(g, initialization='uniform')
g = m.update_graph()
visualize(g, title='Uniform')

m = Model(g, initialization='self_normalized')
g = m.update_graph()
visualize(g, title='Self-Normalized')