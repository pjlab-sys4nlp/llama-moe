import torch
import torch.nn as nn
import torch.nn.init as init


class WeightNorm(nn.Module):
    def __init__(
        self, hidden_size: int, scale: float = 1.0, device=None, dtype=None
    ) -> None:
        super().__init__()

        self.hsz = hidden_size
        self.scale = scale

        self.weight = nn.Parameter(torch.empty(hidden_size, device=device, dtype=dtype))

        self.reset_parameters()

    def reset_parameters(self):
        # init.ones_(self.weight)
        init.constant_(self.weight, self.scale)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        # if torch.isnan(self.weight).any():
        #     remote_breakpoint()
        # return hidden * (self.scale * F.sigmoid(self.weight) + 1.0)
        return hidden * self.weight

    def extra_repr(self) -> str:
        return "hsz={}, scale={}".format(self.hsz, self.scale)
