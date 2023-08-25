import torch
from torch import nn


def backward_hook(module, grad_in, grad_out):
    print(module.name, "grad_in", len(grad_in), [grad_in[i].shape if grad_in[i] is not None else None for i in range(len(grad_in))], grad_in, sep='\n')
    print(module.name, "grad_out", len(grad_out), [grad_out[i].shape if grad_out[i] is not None else None for i in range(len(grad_out))], grad_out, sep='\n')


class Model(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.layer1 = nn.Linear(input_dim, hidden_dim, bias=False)
        self.layer2 = nn.Linear(hidden_dim, output_dim, bias=False)
        self.activation = nn.Sigmoid()

        self.layer1.name = "layer1"
        self.layer2.name = "layer2"

        self.layer1.register_backward_hook(backward_hook)
        self.layer2.register_backward_hook(backward_hook)

    def forward(self, x):
        z1 = self.layer1(x)
        z2 = self.layer2(z1)
        a2 = self.activation(z2)
        return a2


batch_size = 4
input_dim = 128
hidden_dim = 1024
output_dim = 64

model = Model(input_dim, hidden_dim, output_dim)
loss_func = nn.MSELoss()

x = torch.rand((batch_size, input_dim))
target = torch.rand((batch_size, output_dim))

y = model(x)
loss = loss_func(y, target)
loss.backward()

print(model.layer1.weight.grad, model.layer1.weight.grad.shape)
print(model.layer2.weight.grad, model.layer2.weight.grad.shape)
