from abc import abstractmethod
from typing import Any

from torch import nn, tensor


class BaseVAE(nn.Module):
    
    def __init__(self) -> None:
        super(BaseVAE, self).__init__()

    def encode(self, input: tensor) -> list[tensor]:
        raise NotImplementedError

    def decode(self, input: tensor) -> Any:
        raise NotImplementedError

    def sample(self, batch_size: int, current_device: int, **kwargs) -> tensor:
        raise NotImplementedError

    def generate(self, x: tensor, **kwargs) -> tensor:
        raise NotImplementedError

    @abstractmethod
    def forward(self, *inputs: tensor) -> tensor:
        pass

    @abstractmethod
    def loss_function(self, *inputs: Any, **kwargs) -> tensor:
        pass