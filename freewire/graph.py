"""Graph data structure for defining freely wired neural networks.
"""

class Node:
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

class Edge:
  def __init__(self, start, end):
    self.start = start
    self.end = end
    self.weight = 0

class Graph:
  def __init__(self, input_nodes, hidden_nodes, output_nodes):
    edges = []
    for node in hidden_nodes + output_nodes:
      edges += node.in_edges
    self.edges = edges
    self.input_nodes = input_nodes
    self.hidden_nodes = hidden_nodes
    self.output_nodes = output_nodes
    self.nodes = input_nodes + hidden_nodes + output_nodes