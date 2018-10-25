import numpy as np
import torch


class StigmergicModule(torch.nn.Module):
    def __init__(self, inputs=None, outputs=None, *modules, **kwargs):
        torch.nn.Module.__init__(self)
        self.n_inputs = inputs
        self.n_outputs = outputs

    def forward(self, input, **kwargs):
        if len(input.shape) == 1:
            input = input.reshape(1,input.shape[0])
        lastlayer = kwargs["lastlayer"] if "lastlayer" in kwargs else -1
        for i, module in enumerate(self._modules.values()):
            if i == lastlayer:
                break
            input = module(input)
        return input

    def tick(self):
        for layer in self.children():
            if isinstance(layer, StigmergicModule):
                layer.tick()
    
    def reset(self):
        for layer in self.children():
            if isinstance(layer, StigmergicModule):
                layer.reset() 

    def to(self, *args, **kwargs):
        self = torch.nn.Module.to(self, *args, **kwargs)
        for layer in self.children():
            layer.to(*args, **kwargs)
        return self

def Sequential(*modules, **kwargs):
    target = StigmergicModule(**kwargs) if "target" not in kwargs else kwargs["target"]
    for i, module in enumerate(modules):
        target.add_module(str(i), module)
    return target