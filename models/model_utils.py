import torch
import torch.nn.functional as F
from util.constants import Activation


def get_act_func(act):
    """
    Return the corresponding torch activation function given its name
    """
    assert isinstance(act, Activation)
    if act == Activation.ReLU:
        return F.relu  # Used in RP-Paper
    elif act == Activation.Tanh:
        return torch.tanh  # torch.functional.tanh is deprecated
    elif act == Activation.Sigmoid:
        return torch.sigmoid
    else:
        raise NotImplementedError(f"Haven't yet implemented models with {act}, or check the spelling.")

