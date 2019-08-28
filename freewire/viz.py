"""Visualization of graphs.
"""

import os
import secrets
from graphviz import Digraph

def bwr_cmap_hex(val, edge=False):
    """ takes a value from 0 to 1 and returns a color
    inspired from the bwr colormap in matplotlib """
    val = int(val * 255)
    red = int(min(255, 255 + (val - 127.5) * 2))
    blue = int(min(255, 255 + (127.5 - val) * 2))
    green = int(255 - abs(val - 127.5) * 2)

    if edge:
      red = max(0, red - 64)
      green = max(0, green - 64)
      blue = max(0, blue - 64)      
    else:
      red = min(255, red + 64)
      green = min(255, green + 64)
      blue = min(255, blue + 64)

    return "#" + str(hex(red * 0x010000 + \
        green * 0x000100 + \
        blue))[2:].rjust(6,"0")

def visualize(g, title='Graph', inputs=None):
  """Generate a PDF file in `renders` folder with graphviz visualization of
  this graph.

  Capable of visualizing either the parameters of the NN (biases, weights)
  or the activations of the NN dependent on a given input.

  # Arguments:
    title: Name of file to write
    inputs: If set, this will visualize the activations of the network. If left
      as None, this will visualize the parameters of the network.
  """
  
  save_loc = 'renders/{}'.format(title)
  if not os.path.exists('renders/'):
    os.mkdir('renders')
  if os.path.exists(save_loc + '.pdf'):
    os.remove(save_loc +'.pdf')

  dot = Digraph()

  # Labeling with parameters
  node_vals = [node.bias for node in g.input_nodes + g.hidden_nodes + g.output_nodes]
  if abs(max(node_vals)) > abs(min(node_vals)):
    n_max = max(node_vals)
    n_min = max(node_vals) * -1
  elif abs(max(node_vals)) < abs(min(node_vals)):
    n_min = min(node_vals)
    n_max = min(node_vals) * -1
  else:
    n_max = 1
    n_min = -1

  edge_vals = [edge.weight for edge in g.edges]
  if abs(max(edge_vals)) > abs(min(edge_vals)):
    e_max = max(edge_vals)
    e_min = max(edge_vals) * -1
  elif abs(max(edge_vals)) < abs(min(edge_vals)):
    e_min = min(edge_vals)
    e_max = min(edge_vals) * -1
  else:
    e_max = 1
    e_min = 0
  
  for i, node in enumerate(g.input_nodes):
    dot.node(str(id(node)), 
            'Input {}'.format(i),
            shape='doublecircle')

  for node in g.hidden_nodes + g.output_nodes:
    label = ""
    if node.activation == 'linear':
      label += "Σx + {:.2f}".format(node.bias)
    elif node.activation == 'relu':
      label += "max(Σx + {:.2f}, 0)".format(node.bias)
    else:
      label += "{}(Σx + {:.2f})".format(node.activation, node.bias)
    if n_max - n_min == 0:
      fillcolor = 'gray'
    else:
      fillcolor = bwr_cmap_hex((node.bias - n_min) / (n_max - n_min))
    shape = 'circle'
    if node.is_output:
      shape = 'doublecircle'
    dot.node(str(id(node)), 
            label,
            fillcolor=fillcolor,
            style='filled',
            shape=shape
            )

  for edge in g.edges:
    dot.edge(str(id(edge.start)), 
              str(id(edge.end)), 
              label="x{:.2f}".format(edge.weight),
              color=bwr_cmap_hex(
              (edge.weight - e_min) / (e_max - e_min), edge=True))

  
  dot.render(filename=save_loc)
  print("Saved graph visualization to \"{}\"".format(save_loc))
  return dot
