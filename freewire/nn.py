from .graph import Node, Graph

def check_ints(int_list):
  for check in int_list:
    assert isinstance(check, int), "Size value must be a positive integer"
    assert check > 0, "Size value must be a positive integer"


def neuron_level_densenet(input_size, hidden_size, output_size, activation):
  check_ints([input_size, hidden_size, output_size])
  inputs = [Node() for _ in range(input_size)]
  hidden = []
  for _ in range(hidden_size):
    hidden.append(Node(inputs + hidden, activation=activation))
  outputs = [Node(inputs + hidden) for _ in range(output_size)]
  g = Graph(inputs, hidden, outputs)
  return g