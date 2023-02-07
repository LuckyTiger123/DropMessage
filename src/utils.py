import math
from torch import Tensor


def glorot(tensor: Tensor):
    if tensor is not None:
        stdv = math.sqrt(6.0 / (tensor.size(-2) + tensor.size(-1)))
        tensor.data.uniform_(-stdv, stdv)


def zeros(tensor: Tensor):
    if tensor is not None:
        tensor.data.fill_(0)


def ones(tensor: Tensor):
    if tensor is not None:
        tensor.data.fill_(1)


def rand_zero_to_ones(tensor: Tensor):
    if tensor is not None:
        tensor.data.uniform_(0, 1)


def average_agg(tensor: Tensor):
    result = 0
    batch_number = tensor.size(0)
    for i in range(batch_number):
        result += tensor[i]
    result /= batch_number
    return result
