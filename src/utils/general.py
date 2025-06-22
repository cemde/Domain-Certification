import itertools
import sys
from typing import Any, Dict, List
from datetime import datetime
import random
from collections import deque
import uuid
from contextlib import contextmanager

import numpy as np
import torch


def pprint(*args, **kwargs):
    print(f"[{datetime.now()}]", *args, **kwargs)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


class PrintingManager:
    """
    A class to manage the redirection of standard output to a log file. Can be useful if script
    is called from another script and stdout is not redirected properly.

    Attributes:
        log_file (str): The path to the log file. If None, standard output is not redirected.

    Methods:
        __init__(log_file: str = None):
            Initializes the PrintingManager with an optional log file.

        set_stdout(file):
            Redirects standard output to the specified log file.

        reset():
            Resets standard output to the default.
    """

    def __init__(self, log_file: str = None):
        """Initializes the PrintingManager with an optional log file.

        Args:
            log_file (str, optional): The path to the log file. If None, standard output is not redirected.
        """
        self.log_file = log_file
        if log_file is not None:
            self.set_stdout(log_file)

    def set_stdout(self, file):
        sys.stdout = open(self.log_file, "w")

    def reset(self):
        if self.log_file is not None:
            sys.stdout.close()
            sys.stdout = sys.__stdout__


class Timer:
    def __init__(self, maxlen: int = 1_000):
        self.maxlen = maxlen
        self.reset()

    def append(self, time: float):
        self.times.append(time)

    def mean(self):
        return np.mean(self.times)

    def total(self):
        return np.sum(self.times)

    def __str__(self):
        return f"Total time: {self.total():.2f}s, Mean time: {self.mean():.2f}s"

    def reset(self):
        self.times = deque(maxlen=self.maxlen)

    def __repr__(self):
        return self.__str__()

    def __len__(self):
        return len(self.times)


def batched_right_padding(batches: List[torch.Tensor], along_dim: int = 1, fill_value: int = 0) -> torch.Tensor:
    if not along_dim == 1:
        raise NotImplementedError("Only along_dim=1 is supported for now")

    if not batches:
        return torch.tensor([])

    # assert ndim of all tensors in the batch is the same
    ndim = batches[0].ndim
    assert all([tensor.ndim == ndim for tensor in batches])

    # Collect the shapes of the tensors in the batch
    shapes = [tensor.shape for tensor in batches]

    # assert all tensors have the same shape except along_dim
    for dim in range(ndim):
        if dim == along_dim:
            continue
        assert all([shape[dim] == shapes[0][dim] for shape in shapes])

    # get the max size along the along_dim
    max_size = max([shape[along_dim] for shape in shapes])

    # target size of the concatenated tensor
    target_size = list(shapes[0])
    target_size[along_dim] = max_size

    # pad the tensors
    padded_batches = []
    for tensor in batches:
        pad_size = max_size - tensor.shape[along_dim]
        if pad_size > 0:
            new_tensor = torch.full(target_size, fill_value, dtype=tensor.dtype, device=tensor.device)
            slices = [slice(None)] * ndim
            slices[along_dim] = slice(0, tensor.shape[along_dim])
            new_tensor[slices] = tensor
            padded_batches.append(new_tensor)
        else:
            padded_batches.append(tensor)

    return padded_batches


def generate_id() -> str:
    """
    Generate a unique identifier in the format 'cccc-cccc'.

    The identifier consists of 8 characters separated by a hyphen,
    where each character is a hexadecimal digit (0-9 and a-f).

    This function uses Python's `uuid` module to generate a random
    UUID and formats the first 8 characters of its hexadecimal
    representation to match the desired pattern.

    Returns:
        str: A string representing the unique identifier in the format 'cccc-cccc'.

    Example:
        >>> generate_id()
        '1a2b-3c4d'
    """
    # Generate a UUID
    unique_id = uuid.uuid4().hex  # Generates a 32-character hexadecimal string

    # Format it to 'cccc-cccc' where each 'c' is any character
    formatted_id = f"{unique_id[:4]}-{unique_id[4:8]}"

    return formatted_id


# https://stackoverflow.com/questions/5228158/cartesian-product-of-a-dictionary-of-lists/5228294#5228294
def product_dict(**kwargs):
    """
    Generate all possible combinations of key-value pairs from input dictionaries.

    This function takes keyword arguments where each argument is expected to be
    an iterable. It yields dictionaries containing all possible combinations of
    the input values, maintaining the original keys.

    Args:
        **kwargs: Arbitrary keyword arguments. Each argument should be an iterable.

    Yields:
        dict: A dictionary containing a combination of the input values.
            The keys are the same as the input keyword argument names,
            and the values are elements from the corresponding input iterables.

    Example:
        >>> list(product_dict(color=['red', 'blue'], size=['S', 'M']))
        [{'color': 'red', 'size': 'S'},
         {'color': 'red', 'size': 'M'},
         {'color': 'blue', 'size': 'S'},
         {'color': 'blue', 'size': 'M'}]

    Note:
        This function uses `itertools.product` to generate combinations efficiently.
        The order of the yielded dictionaries is determined by the order of the input iterables.
    """
    keys = kwargs.keys()
    for instance in itertools.product(*kwargs.values()):
        yield dict(zip(keys, instance))


def verify_prompt_length_argument(input_string: str) -> bool:
    """Checkss whether the ``input_string`` argument is a valid.

    Args:
        input_string (str): The input string to be checked.

    Returns:
        bool: True if the input string is valid, False otherwise.
    """
    # if not isinstance(input_string, str):
    #     return False
    # Check if the string is "dataset" or "random"
    if input_string in ["dataset", "random"]:
        return True
    # Try to convert the string to an integer
    try:
        int(input_string)
        return True
    except ValueError:
        return False


def supports_bfloat16():
    # Get the compute capability of the current CUDA device
    major, minor = torch.cuda.get_device_capability()

    # bfloat16 is supported on devices with compute capability 8.0 or higher
    # These include NVIDIA Ampere (e.g., A100, RTX 3090) and later architectures
    if major >= 8:
        return True
    else:
        return False


def get_mixed_precision(mixed_precision: bool) -> Dict[str, bool]:
    supports_bf16 = supports_bfloat16()

    if mixed_precision and supports_bf16:
        return {"bf16": True}
    elif mixed_precision and not supports_bf16:
        return {"fp16": True}
    else:
        return {}


@contextmanager
def disable_adapter_layers(model, lora_enabled: bool):
    if lora_enabled:
        model.disable_adapter_layers()
    try:
        yield
    finally:
        if lora_enabled:
            model.enable_adapter_layers()


def move_tree_to_device(tree: dict | list | torch.Tensor, device: torch.device | str) -> dict | list | torch.Tensor:
    """Recursively moves a nested structure of dictionaries, lists, and PyTorch tensors to the specified device."""
    if isinstance(tree, dict):
        return {k: move_tree_to_device(v, device) for k, v in tree.items()}
    elif isinstance(tree, list):
        return [move_tree_to_device(v, device) for v in tree]
    elif isinstance(tree, torch.Tensor):
        return tree.to(device)
    else:
        return tree


def move_tensors_to_cpu(nested_dict: Dict[Any, Any]) -> Dict[Any, Any]:  # TODO remove and use the above.
    """
    Recursively moves all PyTorch tensors in a nested dictionary to CPU memory.

    Args:
        nested_dict (Dict[Any, Any]): The nested dictionary containing various objects
            including potentially PyTorch tensors.

    Returns:
        None: The function modifies the dictionary in place.
    """
    for key, value in nested_dict.items():
        if isinstance(value, dict):  # Recurse into nested dictionaries
            move_tensors_to_cpu(value)
        elif isinstance(value, torch.Tensor):  # Move tensor to CPU if it's a tensor
            nested_dict[key] = value.cpu()
    return nested_dict
