import torch.distributed as dist


def get_world_size():
    if not dist.is_available():
        return 1
    if not dist.is_initialized():
        return 1

    return dist.get_world_size()


def reduce_mean(tensor):
    '''
    Reduce the tensor across all machines, the operation is in-place.
    :param tensor: tensor to reduce
    :return: reduced tensor
    '''
    if not dist.is_available():
        return tensor
    if not dist.is_initialized():
        return tensor

    world_size = get_world_size()
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return tensor.div_(world_size)