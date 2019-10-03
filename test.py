"""Testing module. Run with `pytest test.py`
"""

from freewire import neuron_level_densenet
from freewire import Node, Graph
from freewire import Model

def test_edge():
  n1 = Node()
  n2 = Node(n1)
  assert n2.in_edges[0] == n1.out_edges[0]

def test_graph():
  n1 = Node()
  n2 = Node(n1)
  n3 = Node([n1, n2])
  g = Graph([n1], [n2], [n3])
  assert g.input_nodes[0] == n1
  assert g.hidden_nodes[0] == n2
  assert g.output_nodes[0] == n3
  assert len(g.nodes) == 3

def test_densenet():
  g = neuron_level_densenet(2, 5, 1, 'relu')
  assert len(g.input_nodes) == 2
  assert len(g.hidden_nodes[0].in_edges) == 2
  assert len(g.hidden_nodes[4].out_edges) == 1
  assert len(g.hidden_nodes[2].in_edges) == 4
  assert len(g.output_nodes) == 1
  assert g.hidden_nodes[0].activation == 'relu'

def test_model():
  g = neuron_level_densenet(2, 5, 1, 'relu')
  assert g.input_nodes[0].bias == 0
  assert g.input_nodes[0].out_edges[0].weight == 0
  m = Model(g)
  assert isinstance(g.input_nodes[0].out_edges[0].weight, float)
