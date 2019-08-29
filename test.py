"""Testing module. Run with `pytest test.py`
"""

from freewire.nn import neuron_level_densenet

def test_densenet():
  g = neuron_level_densenet(2, 5, 1, 'relu')
  assert len(g.input_nodes) == 2
  assert len(g.hidden_nodes[0].in_edges) == 2
  assert len(g.hidden_nodes[4].out_edges) == 1
  assert len(g.hidden_nodes[2].in_edges) == 4
  assert len(g.output_nodes) == 1
  assert g.hidden_nodes[0].activation == 'relu'
