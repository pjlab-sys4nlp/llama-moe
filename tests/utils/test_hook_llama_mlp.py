import torch
from torch import nn
from transformers.models.llama.modeling_llama import LlamaMLP


# fmt: off
def backward_hook(module, grad_in, grad_out):
    print(module.name, "grad_in", len(grad_in), [grad_in[i].shape if grad_in[i] is not None else None for i in range(len(grad_in))], grad_in, sep='\n')
    print(module.name, "grad_out", len(grad_out), [grad_out[i].shape if grad_out[i] is not None else None for i in range(len(grad_out))], grad_out, sep='\n')


class Config:
    def __init__(self):
        self.pretraining_tp = 1
        self.hidden_size = 128
        self.intermediate_size = 1024
        self.hidden_act = "silu"


batch_size = 2
seq_len = 4

config = Config()
model = LlamaMLP(config)

model.up_proj.name = "up_proj"
model.gate_proj.name = "gate_proj"
model.down_proj.name = "down_proj"

model.up_proj.register_backward_hook(backward_hook)
model.gate_proj.register_backward_hook(backward_hook)
model.down_proj.register_backward_hook(backward_hook)

loss_func = nn.MSELoss()

x = torch.rand((batch_size * seq_len, config.hidden_size))
target = torch.rand((batch_size * seq_len, config.hidden_size))

# Wrong "grad_in" and "grad_out" will be captured when using inputs with (batch_size, seq_len, *) format !
#################################################################
# x = torch.rand((batch_size, seq_len, config.hidden_size))
# target = torch.rand((batch_size, seq_len, config.hidden_size))
#################################################################

y = model(x)
loss = loss_func(y, target)
loss.backward()

print(model.up_proj.name, "grad", model.up_proj.weight.grad, model.up_proj.weight.grad.shape, sep='\n')
print(model.gate_proj.name, "grad", model.gate_proj.weight.grad, model.gate_proj.weight.grad.shape, sep='\n')
print(model.down_proj.name, "grad", model.down_proj.weight.grad, model.down_proj.weight.grad.shape, sep='\n')
