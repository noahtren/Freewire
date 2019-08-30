"""Functions used directly by model.
Activation functions, initializers, and more.
"""

import torch
import torch.nn.functional as F

activation_map = {
  'relu': F.relu,
  'selu': F.selu,
  'linear':lambda x: x, # identity
  'leaky':F.leaky_relu,
  'sigmoid':F.sigmoid,
  'softmax':F.softmax
}

def he_initialization(input_indices):
  """Neuron-level He initialization
  """
  # prev_sizes is number of nodes in previous layer per output neuron
  # this allows for He initialization per neuron, since
  # input layer is not homogeneous
  prev_sizes = (input_indices!=0).sum(dim=1).cpu()
  prev_sizes = torch.unsqueeze(prev_sizes, 1).type(torch.float)
  scales = (2 / prev_sizes) ** (1/2)
  weights = torch.randn(input_indices.shape) * scales
  del prev_sizes
  return weights

def uniform(input_indices):
  """Uniform distribution from -1 to 1
  """
  weights = (torch.rand(input_indices.shape).cuda() * 2) - 1
  return weights

initialization_map = {
  'He':he_initialization,
  'uniform':uniform
}
