import os
from typing import Tuple
import rich
from datetime import datetime
import builtins

import torch
import torch.distributed as dist


def local_rank():
    return int(os.environ.get("LOCAL_RANK") or 0)


def is_main_process():
    return int(os.environ.get("LOCAL_RANK") or 0) == 0


def set_up_ddp():
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    if ddp:
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
    else:
        device_map = "auto"

    return ddp, world_size, device_map


def print(*args, **kwargs):
    if is_main_process():
        builtins.print(*args, **kwargs)


def pprint(*args, **kwargs):
    if is_main_process():
        rich.print(f"[{datetime.now()}]", *args, **kwargs)


def pprint_all_rank(*args, **kwargs):
    rich.print(f"[{datetime.now()}][Rank: {local_rank()}]", *args, **kwargs)


def _send_str(data: str) -> None:
    # encode strings to ints
    data_numeric = []
    for d in data:
        data_numeric.append([ord(c) for c in d])

    lengths = [len(d) for d in data_numeric]

    # pad data and create tensor
    max_length = max(lengths)
    data_numeric_padded = []
    for d in data_numeric:
        d_padded = d + [-1] * (max_length - len(d))
        data_numeric_padded.append(d_padded)
    data_numeric_tensor = torch.tensor(data_numeric_padded, dtype=torch.int64, device=f"cuda:{local_rank()}")
    data_numeric_tensor_shape = torch.tensor(data_numeric_tensor.shape, dtype=torch.int64, device=f"cuda:{local_rank()}")

    # broadcast the shape of the tensor containing the data
    dist.broadcast(data_numeric_tensor_shape, src=local_rank())
    # broadcast the tensor containing the data
    dist.broadcast(data_numeric_tensor, src=local_rank())


def _receive_str(src: int) -> Tuple[str, ...]:
    # receive the shape of the tensor containing the data
    data_numeric_tensor_shape = torch.tensor([0, 0], dtype=torch.int64, device=f"cuda:{local_rank()}")
    dist.broadcast(data_numeric_tensor_shape, src=src)

    # receive the tensor containing the data
    data_numeric_tensor_shape = tuple(data_numeric_tensor_shape.cpu().tolist())
    data_numeric_tensor = torch.zeros(data_numeric_tensor_shape, dtype=torch.int64, device=f"cuda:{local_rank()}")
    dist.broadcast(data_numeric_tensor, src=src)

    # decode the tensor to strings
    data_numeric = data_numeric_tensor.tolist()

    data = []
    for d in data_numeric:
        d = [c for c in d if c != -1]
        d = "".join([chr(c) for c in d])
        data.append(d)

    return tuple(data)


def broadcast_str(*data, source: int = 0) -> Tuple[str, ...]:
    """
    Broadcasts a tuple of strings to all processes in a distributed environment.

    Args:
        *data: Variable number of strings to be broadcasted.
        source (int): The rank of the process that holds the original data. Defaults to 0.

    Returns:
        Tuple[str, ...]: A tuple containing the broadcasted strings.

    Raises:
        AssertionError: If any of the input data is not a string.

    """
    # assert all data are string
    for d in data:
        assert isinstance(d, str)

    # convert data to list of lists of ints
    if local_rank() == source:
        _send_str(data)
    else:
        data = _receive_str(source)

    return data
