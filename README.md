# CGDs
## Overview
`CGDs` is a package implementing optimization algorithms including three variants of [CGD](https://arxiv.org/abs/1905.12103)  in [Pytorch](https://pytorch.org/) with Hessian vector product and conjugate gradient.  
`CGDs` is for competitive optimization problem such as generative adversarial networks (GANs) as follows: 
$$
\min_{\mathbf{x}}f(\mathbf{x}, \mathbf{y}) \min_{\mathbf{y}} g(\mathbf{x}, \mathbf{y})
$$

**Update**: ACGD now supports distributed training. Set `backward_mode=True` to enable. We have new member GMRES-ACGD that can work for general two-player competitive optimization problems.

## Installation 
CGDs can be installed with the following pip command. It requires Python 3.6+.
```bash
pip3 install CGDs
```
You can also directly download the `CGDs` directory and copy it to your project.

## Package description

The `CGDs` package implements the following optimization algorithms with Pytorch:

- `BCGD` : CGD algorithm in [Competitive Gradient Descent](https://arxiv.org/abs/1905.12103).
- `ACGD` : ACGD algorithm in [Implicit competitive regularization in GANs](https://arxiv.org/abs/1910.05852). 
- `GACGD`: GMRES-ACGD that works for general 
## How to use
Quickstart with notebook: [Examples of using ACGD](https://colab.research.google.com/drive/1-52aReaBAPNBtq2NcHxKkVIbdVXdyqtH?usp=sharing). 

Similar to Pytorch package `torch.optim`, using optimizers in `CGDs` has two main steps: construction and update steps. 
### Construction
To construct an optimizer, you have to give it two iterables containing the parameters (all should be `Variable`s). 
Then you need to specify the `device`, `learning rate`s. 

Example:
```python

from src import CGDs
import torch
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
optimizer = CGDs.ACGD(max_param=model_G.parameters(), min_params=model_D.parameters(), 
                      lr_max=1e-3, lr_min=1e-3, device=device)
optimizer = CGDs.BCGD(max_params=[var1, var2], min_params=[var3, var4, var5], 
                      lr_max=0.01, lr_min=0.01, device=device)   
```

### Update step 

Both two optimizers have `step()` method, which updates the parameters according to their update rules. The function can be called once the computation graph is created. You have to pass in the loss but do not have to compute gradients before `step()` , which is *different* from `torch.optim`.

Example:

```python
for data in dataset:
    optimizer.zero_grad()
    real_output = model_D(data)
   	latent = torch.randn((batch_size, latent_dim), device=device)
    fake_output = D(G(latent))
    loss = loss_fn(real_output, fake_output)
    optimizer.step(loss=loss)
```
For general competitive optimization, two losses should be defined and passed to optimizer.step
```python
loss_x = loss_f(x, y)
loss_y = loss_g(x, y)
optimizer.step(loss_x, loss_y)
```

## Citation

Please cite it if you find this code useful. 

```latex
@misc{cgds-package,
  author = {Hongkai Zheng},
  title = {CGDs},
  year = {2020},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/devzhk/cgds-package}},
}
```
