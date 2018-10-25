import torch
from torch.nn.parameter import Parameter

from .stigmergic_layer import StigmergicLayer

class SimpleLayer(StigmergicLayer):
    def __init__(self, inputs, outputs, **kwargs):
        StigmergicLayer.__init__(self, inputs, outputs, **kwargs)
        self.reset()
        self.markval = Parameter(torch.randn(1, outputs))
        self.tickval = Parameter(torch.randn(1, outputs))
        
    
    def forward(self, input):
        ret = self.activationFunction(input.matmul(self.weights) - self.th + self.bias)
        self.th = self.clamp(self.th + ret*self.markval)
        return ret
    
    def tick(self):
        self.th = self.clamp(self.th-self.tickval)

    def reset(self):
        self.th = self.clamp(self.init_th)
        self.weights = self.init_weights*1


    def to(self, *args, **kwargs):
        self = StigmergicLayer.to(self, *args, **kwargs) 
        self.markval = self.markval.to(*args, **kwargs)
        self.tickval = self.tickval.to(*args, **kwargs) 
        
        self.weights = self.weights.to(*args, **kwargs)
        self.th = self.th.to(*args, **kwargs)

        return self