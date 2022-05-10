import torch


def CG(Avp,                 # linear operator
       b,                   # RHS of the linear system
       x0=None,             # initial guess
       max_iter=None,       # maximum number of iterations
       tol=1e-5,            # relative tolerance
       atol=1e-6,           # absolute tolerance
       track=False,         # track the residual error
       ):
    if max_iter is None:
        max_iter = b.shape[0]
    if x0 is None:
        x = torch.zeros_like(b)
        r = b.detach().clone()
    else:
        Av = Avp(x0)
        r = b.detach().clone() - Av
        x = x0

    p = r.clone()
    rdotr = torch.dot(r, r)
    bdotb = torch.dot(b, b)
    residual_tol = max(tol * tol * bdotb, atol * atol)
    err_history = []

    if track:
        err_history.append(torch.sqrt(rdotr / bdotb).item())

    if rdotr < residual_tol:
        return x, (0, err_history)
    # main body of CG
    for i in range(max_iter):
        Ap = Avp(p)

        alpha = rdotr / torch.dot(p, Ap)
        x.add_(alpha * p)
        r.add_(-alpha * Ap)
        new_rdotr = torch.dot(r, r)
        beta = new_rdotr / rdotr
        p = r + beta * p
        rdotr = new_rdotr
        if track:
            err_history.append(torch.sqrt(rdotr / bdotb).item())
        if rdotr < residual_tol:
            break

    if i == max_iter - 1:
        print(f'Squared relative residual error after {max_iter} iterations of CG: {rdotr / residual_tol}')
    return x, (i + 1, err_history)
