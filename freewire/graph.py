"""Graph data structure for defining freely wired neural networks.
"""

class Node:
  """Node in freely wired neural network.
  # Arguments
    inputs: a list of input nodes. Pass no argument if this is an input node
    output_index: optionally specify index of output node in flattened output tensor
    activation: activation function. Options are defined in freewire/model_utils.py
    op_id: optional operation id, used to force Nodes to exist in the same operation
  """
  def __init__(self, inputs=[], output_index=-1, activation='linear', op_id='any'):
    if isinstance(inputs, Node):
      inputs = [inputs]
    self.inputs = inputs
    self.output_index = output_index
    self.op_id = op_id

    # infer node type
    self.is_input = False
    self.is_output = False
    self.is_hidden = False
    if self.inputs == []:
      self.is_input = True
    elif output_index >= 0:
      self.is_output = True
    else:
      self.is_hidden = True

    # activation function
    if self.is_input:
      assert activation == 'linear', "Input node can't have an activation function"
    self.activation = activation

    # add edges to global list, if applicable
    self.out_edges = []
    self.in_edges = []
    for inp in self.inputs:
      new_edge = Edge(inp, self)
      inp.out_edges.append(new_edge)
      self.in_edges.append(new_edge)

    # attributes to be set later
    self.tape_index = -1
    self.assigned = False
    self.bias = 0
    self.grad = 0

class Edge:
  def __init__(self, start, end):
    self.start = start
    self.end = end
    self.weight = 0
    self.grad = 0

class Graph:
  """Graph representing topology of freely wired neural network.
  # Arguments
    input_nodes: list of input nodes
    hidden_nodes: list of hidden nodes, not necessarily ordered (although it does
      make compilation more efficient)
    output_nodes: list of output nodes
  """
  def __init__(self, input_nodes, hidden_nodes, output_nodes):
    assert isinstance(input_nodes, list)
    assert isinstance(hidden_nodes, list)
    assert isinstance(output_nodes, list)
    edges = []
    for node in hidden_nodes + output_nodes:
      edges += node.in_edges
    self.edges = edges
    self.input_nodes = input_nodes
    self.hidden_nodes = hidden_nodes
    self.output_nodes = output_nodes
    self.nodes = input_nodes + hidden_nodes + output_nodes
    for node in self.output_nodes:
      node.is_output = True