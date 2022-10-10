import functools

import numpy as np
import torch


def is_tensor_like(x):
    return torch.is_tensor(x) or type(x) is np.ndarray


def handle_batch_input(n):
    def _handle_batch_input(func):
        """For func that expect 2D input, handle input that have more than 2 dimensions by flattening them temporarily"""

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # assume inputs that are tensor-like have compatible shapes and is represented by the first argument
            batch_dims = []
            for arg in args:
                if is_tensor_like(arg) and len(arg.shape) > n:
                    batch_dims = arg.shape[:-(n - 1)]  # last dimension is type dependent; all previous ones are batches
                    break
            # no batches; just return normally
            if not batch_dims:
                return func(*args, **kwargs)

            # reduce all batch dimensions down to the first one
            args = [v.view(-1, *v.shape[-(n - 1):]) if (is_tensor_like(v) and len(v.shape) > 2) else v for v in args]
            ret = func(*args, **kwargs)
            # restore original batch dimensions; keep variable dimension (nx)
            if type(ret) is tuple:
                ret = [v if (not is_tensor_like(v) or len(v.shape) == 0) else (
                    v.view(*batch_dims, *v.shape[-(n - 1):]) if len(v.shape) == n else v.view(*batch_dims)) for v in
                       ret]
            else:
                if is_tensor_like(ret):
                    if len(ret.shape) == n:
                        ret = ret.view(*batch_dims, *ret.shape[-(n - 1):])
                    else:
                        ret = ret.view(*batch_dims)
            return ret

        return wrapper

    return _handle_batch_input