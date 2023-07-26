from torch import nn

valid_kernel_name = "linear"


def get_kernel(kernel_name):
    if kernel_name == "linear":
        return LinearKernel()
    else:
        raise ValueError(
            "Invalid kernel name, expected "
            + str(valid_kernel_name)
            + ", get "
            + kernel_name
            + "."
        )


class LinearKernel(nn.Module):
    def __init__(self):
        super(LinearKernel, self).__init__()

    def forward(self, weight):
        return weight
