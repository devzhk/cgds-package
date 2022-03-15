import torch
from torch.utils.dlpack import to_dlpack, from_dlpack

import numpy as np
import cupy as cp

from cupyx.scipy.sparse.linalg import gmres
from cupyx.scipy.sparse.linalg import LinearOperator

from functools import partial
from .cgd_utils import Hvp_vec


def MvProd(vector,     # cupy tensor
           grad_fy,
           grad_gx,
           x_params,
           y_params,
           lr_x, lr_y,
           trigger=None,
           x_reducer=None,
           y_reducer=None,
           rebuild=False):
    '''
    Compute matrix vector product:
    \begin{pmatrix}
        I_m                                               & \eta_x \frac{\partial^2f}{\partial x \partial y}f \\
        \eta_y \frac{\partial^2g}{\partial y \partial x}g & I_n
    \end{pmatrix}
    \begin{pmatrix}
        b1 \\
        b2
    \end{pmatrix}

    Return:
        p1, p2
        where p1 =
    '''
    vec = from_dlpack(vector.toDlpack())
    len_x = lr_x.shape[0]
    len_y = lr_y.shape[0]
    v1 = vec[0: len_x]
    v2 = vec[len_x: len_x + len_y]
    h1 = Hvp_vec(grad_fy, x_params, v2,
                 retain_graph=True, trigger=trigger,
                 reducer=x_reducer,
                 rebuild=rebuild)
    p1 = v1 + lr_x * h1
    h2 = Hvp_vec(grad_gx, y_params, v1,
                 retain_graph=True, trigger=trigger,
                 reducer=y_reducer,
                 rebuild=rebuild)
    p2 = v2 + lr_y * h2
    res = torch.cat([p1, p2])
    return cp.asarray(res)


def CuGMRES(grad_fy,  # gradient vector, \nabla_{y}f
            grad_gx,  # gradient vector, \nabla_{x}g
            x_params, y_params,  # parameters of x and y
            b,
            # tuple of two vectors, RHS of the linear system in which the first half has the same shape as grad_gx, the second half has the same shape as grad_fy
            lr_x, lr_y,  # adaptive learning rate vectors for x_params and y_params
            x0=None,  # initial guess, tuple has the same shape as b
            max_iter=None,
            tol=1e-8,
            trigger=None,
            x_reducer=None,
            y_reducer=None,
            rebuild=False):
    '''
    GMRES solver to solve
    $$
    \begin{pmatrix}
        \Delta x \\
        \Delta y
    \end{pmatrix} =
    \begin{pmatrix}
        I_m                                               & \eta_x \frac{\partial^2f}{\partial x \partial y}f \\
        \eta_y \frac{\partial^2g}{\partial y \partial x}g & I_n
    \end{pmatrix}^{-1}
    \begin{pmatrix}
        b1 \\
        b2
    \end{pmatrix}
    $$
    '''
    # bnorm = torch.norm(b)
    #
    # # if no inverse, it degrades to Adam
    # if max_iter == 0 or bnorm < 1e-8:
    #     return b
    #
    # if x0 is None:
    #     x0 = torch.zeros_like(b)
    # define matrix vector product
    rhs = cp.asarray(b)
    Avp = partial(MvProd,
                  grad_fy=grad_fy, grad_gx=grad_gx,
                  x_params=x_params, y_params=y_params,
                  lr_x=lr_x, lr_y=lr_y,
                  trigger=trigger,
                  x_reducer=x_reducer, y_reducer=y_reducer,
                  rebuild=rebuild)
    length = lr_x.shape[0] + lr_y.shape[0]
    LinOp = LinearOperator((length, length), matvec=Avp, dtype=np.float32)
    sol, info = gmres(LinOp, rhs, tol=tol, restart=max_iter, maxiter=max_iter)
    sol = from_dlpack(sol.toDlpack())
    return sol
