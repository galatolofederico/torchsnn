import torch
from torch.nn.parameter import Parameter

from .stigmergic_layer import StigmergicLayer

class FullLayer(StigmergicLayer):
    def __init__(self, inputs, outputs, **kwargs):
        StigmergicLayer.__init__(self, inputs, outputs, **kwargs)
        self.reset()
        self.markval = Parameter(torch.randn(1, outputs))
        self.tickval = Parameter(torch.randn(1, outputs))

        self.weights_markval = Parameter(torch.randn(inputs, outputs))
        self.weights_tickval = Parameter(torch.randn(inputs, outputs))
        
    
    def forward(self, input):
        ret = self.activationFunction(input.matmul(self.weights) - self.th + self.bias)
        self.th = self.clamp(self.th + ret*self.markval)
        self.weights = self.clamp(self.weights + (input.view(-1,1,input.shape[1])*self.weights_markval.t()).permute(0,2,1)[0] - self.weights_tickval)
        return ret
    
    def tick(self):
        self.th = self.clamp(self.th-self.tickval)
        self.weights = self.clamp(self.weights-self.weights_tickval)

    def reset(self):
        self.th = self.clamp(self.init_th)
        self.weights = self.clamp(self.init_weights)

    def to(self, *args, **kwargs):
        self = StigmergicLayer.to(self, *args, **kwargs) 
        self.markval = self.markval.to(*args, **kwargs)
        self.tickval = self.tickval.to(*args, **kwargs) 
        self.weights_markval = self.weights_markval.to(*args, **kwargs) 
        self.weights_tickval = self.weights_tickval.to(*args, **kwargs) 

        self.weights = self.weights.to(*args, **kwargs)
        self.th = self.th.to(*args, **kwargs)

        return self
