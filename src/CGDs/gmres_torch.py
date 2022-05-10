import torch


def _check_nan(vec, msg):
    if torch.isnan(vec).any():
        raise ValueError(msg)


def _safe_normalize(x, threshold=None):
    norm = torch.norm(x)
    if threshold is None:
        threshold = torch.finfo(norm.dtype).eps
    normalized_x = x / norm if norm > threshold else torch.zeros_like(x)
    return normalized_x, norm


def arnoldi(vec,  # Matrix vector product
            V,  # List of existing basis
            H,  # H matrix
            j):  # number of basis
    '''
    Arnoldi iteration to find the j th l2-orthonormal vector
    compute the j-1 th column of Hessenberg matrix
    '''
    _check_nan(vec, 'Matrix vector product is Nan')

    for i in range(j):
        H[i, j - 1] = torch.dot(vec, V[i])
        vec = vec - H[i, j-1] * V[i]
    new_v, vnorm = _safe_normalize(vec)
    H[j, j - 1] = vnorm
    return new_v


def cal_rotation(a, b):
    '''
    Args:
        a: element h in position j
        b: element h in position j+1

    Returns:
        cosine = a / \sqrt{a^2 + b^2}
        sine = - b / \sqrt{a^2 + b^2}
    '''
    c = torch.sqrt(a * a + b * b)
    return a / c, - b / c


def apply_given_rotation(H, cs, ss, j):
    '''
    Apply givens rotation to H column
    :param H:
    :param cs:
    :param ss:
    :param j:
    :return:
    '''
    # apply previous rotation to the 0->j-1 columns
    for i in range(j):
        tmp = cs[i] * H[i, j] - ss[i] * H[i + 1, j]
        H[i + 1, j] = cs[i] * H[i+1, j] + ss[i] * H[i, j]
        H[i, j] = tmp
    cs[j], ss[j] = cal_rotation(H[j, j], H[j + 1, j])
    H[j, j] = cs[j] * H[j, j] - ss[j] * H[j + 1, j]
    H[j + 1, j] = 0
    return H, cs, ss


def GMRES(Avp,              # Linear operator
          b,                # RHS of the linear system
          x0=None,          # initial guess, tuple has the same shape as b
          max_iter=None,    # maximum number of iterations
          tol=1e-6,         # relative tolerance
          atol=1e-6,        # absolute tolerance
          track=False       # If True, keep a history of the relative residual error
          ):
    bnorm = torch.norm(b)
    _check_nan(b, 'RHS of the system is Nan')
    if max_iter == 0 or bnorm < 1e-8:
        return b, (0, 0)

    if max_iter is None:
        max_iter = b.shape[0]

    if x0 is None:
        x0 = torch.zeros_like(b)
        r0 = b
    else:
        r0 = b - Avp(x0)

    new_v, rnorm = _safe_normalize(r0)
    # initial guess residual
    beta = torch.zeros(max_iter + 1, device=b.device)
    beta[0] = rnorm

    err_history = []
    if track:
        err_history.append((rnorm / bnorm).item())

    V = []
    V.append(new_v)
    H = torch.zeros((max_iter + 1, max_iter + 1), device=b.device)
    cs = torch.zeros(max_iter, device=b.device)  # cosine values at each step
    ss = torch.zeros(max_iter, device=b.device)  # sine values at each step

    for j in range(max_iter):
        p = Avp(V[j])
        new_v = arnoldi(p, V, H, j + 1)  # Arnoldi iteration to get the j+1 th ba
        # sis
        V.append(new_v)

        H, cs, ss = apply_given_rotation(H, cs, ss, j)
        beta[j + 1] = ss[j] * beta[j]
        beta[j] = cs[j] * beta[j]
        residual = torch.abs(beta[j + 1])
        if track:
            err_history.append((residual / bnorm).item())
        if residual < tol * bnorm or residual < atol:
            # print(f'\nGMRES iterations: {j}')
            break
        # print(f'{j}the gmres iteration:  residual {residual/bnorm}')
    if j == max_iter - 1:
        print(f'\nMax number of iterations: {max_iter}')
    y, _ = torch.triangular_solve(beta[0:j + 1].unsqueeze(-1), H[0:j + 1, 0:j + 1])  # j x j
    V = torch.stack(V[:-1], dim=0)
    sol = x0 + V.T @ y.squeeze(-1)
    return sol, (j, err_history)