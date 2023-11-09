import torch
from torch import nn
from cv import Preproc


class GestureNet(nn.Module):
    def __init__(self) -> None:
        super(GestureNet, self).__init__()
