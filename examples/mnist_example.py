import random

import mnist
import numpy as np
import torch

from freewire import Model
from freewire import Node, Graph

"""Data prep
"""

X_tr = mnist.train_images()
y_tr = mnist.train_labels()

X_val = mnist.test_images()
y_val = mnist.test_labels()

X_tr = np.reshape(X_tr, (60000, 784))
X_val = np.reshape(X_val, (10000, 784))

X_tr = torch.tensor(X_tr, dtype=torch.float).cuda()
X_tr = X_tr / X_tr.max()
y_tr = torch.tensor(y_tr, dtype=torch.long).cuda()
X_val = torch.tensor(X_val, dtype=torch.float).cuda()
X_val = X_val / X_val.max()
y_val = torch.tensor(y_val, dtype=torch.long)

"""Model creation
"""

inputs = [Node() for _ in range(0, 784)]
hidden = [Node(inputs, activation='selu') for _ in range(0, 200)]
hidden2 = [Node(random.choices(hidden, k=100), activation='selu') for _ in range(0, 50)]
outputs = [Node(hidden + hidden2, activation='softmax') for _ in range(10)]
g = Graph(inputs, hidden + hidden2, outputs)
m = Model(g, initialization='he')
m.compile('adam', 'crossentropy')
m.fit(X_tr, y_tr, epochs=10, batch_size=128)

preds = []
for i in range(0, 20):
  preds.append(m(X_val[i*500:(i+1)*500]).cpu().detach())

pred = torch.cat(preds, axis=0)
label_pred = pred.argmax(axis=1)
correct = (label_pred == y_val).nonzero().shape[0]
print("Final accuracy: {}/10000".format(correct))