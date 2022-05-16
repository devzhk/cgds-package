import torch
import torch.nn as nn
import torch.utils.data as data
import random
import numpy as np
import time

from CGDs import ACGD

random.seed(1234)
np.random.seed(1234)
torch.manual_seed(1234)
torch.cuda.manual_seed_all(1234)


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('=======================ACGD=========================')


D = nn.Linear(2, 1, bias=True).to(device)
G = nn.Linear(1, 2, bias=True).to(device)
nn.init.constant_(D.weight, 2.0)
nn.init.constant_(G.weight, 4.0)
nn.init.constant_(D.bias, -1.0)
nn.init.constant_(G.bias, 1.0)

optimizer2 = ACGD(G.parameters(), D.parameters(),
                  lr_max=1.0, lr_min=1.0, tol=1e-10)

# torch.cuda.synchronize()
start = time.time()

for i in range(5):
    z = torch.ones(1, device=device)
    loss = D(G(z))
    optimizer2.step(loss)
    print('===D weight :{}====='.format(D.weight.data))
    print(f'===D bias: {D.bias.data}===')
    print('===G weight: {}====='.format(G.weight.data))
    print(f'===G bias {G.bias.data}===')
    print()

# torch.cuda.synchronize()
end = time.time()
print(f'Time cost: {end - start}')


