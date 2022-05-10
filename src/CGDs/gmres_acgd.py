'''
Generalized ACGD that works for general sum game
'''
import math
import torch
import torch.autograd as autograd

from .gmres_torch import GMRES
from .cgd_utils import zero_grad, Hvp_vec, vectorize_grad

from functools import partial


def MvProd(vec,  # vector
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
    return torch.cat([p1, p2])


class GACGD(object):
    def __init__(self, x_params, y_params,
                 x_reducer=None, y_reducer=None,
                 lr_x=1e-3, lr_y=1e-3,
                 eps=1e-6, beta=0.99,
                 tol=1e-5, atol=1e-6,
                 max_iter=None,
                 track_cond=None):  # function to check if the optimizer track the inner state of iterative solver
        self.x_reducer = x_reducer
        self.y_reducer = y_reducer
        self.x_params = list(x_params)
        self.y_params = list(y_params)
        self.max_iter = max_iter
        self.state = {'lr_x': lr_x, 'lr_y': lr_y,
                      'eps': eps,
                      'tol': tol, 'atol': atol,
                      'beta': beta, 'step': 0,
                      'x0': None,
                      'sq_exp_avg_x': None, 'sq_exp_avg_y': None}
        self.info = {'ada_lr_x': None, 'ada_lr_y': None,
                     'errors': 0, 'iter_num': 0}
        self.track_cond = track_cond

    def zero_grad(self):
        zero_grad(self.x_params)
        zero_grad(self.y_params)

    def get_info(self):
        if self.info['grad_x'] is None:
            print('Warning! No update information stored. Set collect_info=True before call this method')
        return self.info

    def state_dict(self):
        return self.state

    def load_state_dict(self, state_dict, rank=None):
        keys = ['old_x', 'old_y', 'sq_exp_avg_max', 'sq_exp_avg_min']
        if rank:
            for key in keys:
                if key in state_dict:
                    state_dict[key] = state_dict[key].to(rank)
        self.state.update(state_dict)
        print('Load state: {}'.format(state_dict))

    def set_lr(self, lr_x, lr_y):
        self.state.update({'lr_x': lr_x, 'lr_y': lr_y})
        print('Maximizing side learning rate: {:.4f}\n '
              'Minimizing side learning rate: {:.4f}'.format(lr_x, lr_y))

    def step(self, loss_x, loss_y, trigger=None):
        '''
        Update model weights
        :param loss_x: objective function of x parameters
        :param loss_y: objective function of y parameters
        :param trigger: dummy loss to trigger DDP gradient reduction w.r.t. model parameters
        :return:
        '''
        if trigger is None:
            trigger = torch.zeros(1, device=loss_x.device)
        lr_x = self.state['lr_x']
        lr_y = self.state['lr_y']
        beta = self.state['beta']
        eps = self.state['eps']
        tol = self.state['tol']
        atol = self.state['atol']
        time_step = self.state['step'] + 1
        self.state['step'] = time_step
        should_rebuild = time_step < 2

        if hasattr(self.track_cond, '__call__'):
            track_flag = self.track_cond(loss_x, time_step)
        else:
            track_flag = False

        autograd.backward(loss_x, retain_graph=True, inputs=self.x_params)
        grad_x = vectorize_grad(self.x_params)

        autograd.backward(loss_y, retain_graph=True, inputs=self.y_params)
        grad_y = vectorize_grad(self.y_params)

        sq_avg_x = self.state['sq_exp_avg_x']
        sq_avg_y = self.state['sq_exp_avg_y']
        sq_avg_x = torch.zeros_like(grad_x, requires_grad=False) if sq_avg_x is None else sq_avg_x
        sq_avg_y = torch.zeros_like(grad_y, requires_grad=False) if sq_avg_y is None else sq_avg_y
        sq_avg_x.mul_(beta).addcmul_(grad_x, grad_x, value=1 - beta)
        sq_avg_y.mul_(beta).addcmul_(grad_y, grad_y, value=1 - beta)

        bias_correction = 1 - beta ** time_step
        lr_x = math.sqrt(bias_correction) * lr_x / sq_avg_x.sqrt().add(eps)
        lr_y = math.sqrt(bias_correction) * lr_y / sq_avg_y.sqrt().add(eps)

        scaled_grad_x = lr_x * grad_x
        scaled_grad_y = lr_y * grad_y
        RHS = torch.cat([scaled_grad_x, scaled_grad_y])

        grad_fy = autograd.grad(loss_x, self.y_params, create_graph=True)
        grad_fy_vec = torch.cat([g.contiguous().view(-1) for g in grad_fy])
        grad_gx = autograd.grad(loss_y, self.x_params, create_graph=True)
        grad_gx_vec = torch.cat([g.contiguous().view(-1) for g in grad_gx])

        # prev_x0 = self.state['x0']
        prev_x0 = None
        Avp = partial(MvProd,
                      grad_fy=grad_fy_vec, grad_gx=grad_gx_vec,
                      x_params=self.x_params, y_params=self.y_params,
                      lr_x=lr_x, lr_y=lr_y,
                      trigger=trigger,
                      x_reducer=self.x_reducer, y_reducer=self.y_reducer,
                      rebuild=should_rebuild)
        soln, (num_iter, err_history) = GMRES(Avp=Avp, b=RHS, x0=prev_x0,
                                              max_iter=self.max_iter,
                                              tol=tol, atol=atol,
                                              track=track_flag)

        self.state['x0'] = soln
        self.state.update(
            {
                'sq_exp_avg_x': sq_avg_x,
                'sq_exp_avg_y': sq_avg_y
            }
        )
        # track statistics
        if track_flag:
            self.info.update(
                {
                    'ada_lr_x': lr_x.cpu().numpy(),
                    'ada_lr_y': lr_y.cpu().numpy(),
                    'num_iter': num_iter,
                    'errors': err_history
                }
            )

        # update parameters
        with torch.no_grad():
            index = 0
            for p in self.x_params:
                size = p.numel()
                p.data.add_(- soln[index:index + size].reshape(p.shape))
                index += size
            for p in self.y_params:
                size = p.numel()
                p.data.add_(- soln[index:index + size].reshape(p.shape))
                index += size
