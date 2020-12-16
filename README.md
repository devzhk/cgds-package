# CGDs
## Overview
`CGDs` is a package implementing optimization algorithms including [CGD](https://arxiv.org/abs/1905.12103) and [ACGD](https://arxiv.org/abs/1910.05852)  in [Pytorch](https://pytorch.org/) with Hessian vector product and conjugate gradient.  
`CGDs` is for minimax optimization problem such as generative adversarial networks (GANs) as follows: 
$$
\min_{\mathbf{x}} \max_{\mathbf{y}} f(\mathbf{x}, \mathbf{y})
$$

**Warning**: This implementation is only for zero sum game setting because it relies on conjugate gradient method to solve matrix inversion efficiently, which requires the matrix to be positive definite. If you are using competitive gradient descent (CGD) algorithm for non-zero sum games, please check more details in CGD paper https://arxiv.org/abs/1905.12103. For example, GMRES (the generalized minimal residual) algorithm can be a solver for non-zero sum setting. 
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

## How to use
Similar to Pytorch package `torch.optim`, using optimizers in `CGDs` has two main steps: construction and update steps. 
### Construction
To construct an optimizer, you have to give it two iterables containing the parameters (all should be `Variable`s). 
Then you need to specify the `device`, `learning rate`s. 

Example:
```python
import CGDs
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
## Changelog
- version 0.0.2: adjust the stopping criterion of CG for better stability


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