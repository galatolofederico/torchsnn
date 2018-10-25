import torch
from torch.nn.parameter import Parameter
from ..stigmergic_module import StigmergicModule

class StigmergicLayer(StigmergicModule):
    def __init__(self, inputs, outputs, **kwargs):
        StigmergicModule.__init__(self, inputs, outputs, **kwargs)
        
        self.clamp_min = 0 if "clamp_min" not in kwargs else kwargs["clamp_min"]
        self.clamp_max = None if "clamp_max" not in kwargs else kwargs["clamp_max"]

        self.init_weights = Parameter(torch.randn(inputs, outputs)) if "init_weights" not in kwargs else kwargs["weights"]
        self.bias = Parameter(torch.randn(1, outputs)) if "bias" not in kwargs else kwargs["bias"]
        
        self.init_th = Parameter(torch.randn(1, outputs)) if "init_th" not in kwargs else kwargs["perc_init_th"]

    def clamp(self, x):
        if self.clamp_min is not None and self.clamp_max is not None:
            return self.clamp_min + (self.clamp_max - self.clamp_min)*torch.sigmoid((x-self.clamp_min)/(self.clamp_max-self.clamp_min))
        elif self.clamp_min is not None and self.clamp_max is None:
            return self.clamp_min + torch.nn.functional.softplus(x-self.clamp_min)
        elif self.clamp_min is None and self.clamp_max is not None:
            return -(self.clamp_max + torch.nn.functional.softplus(-x+self.clamp_max))
        else:
            return 1*x
            
    def activationFunction(self, x):
        return torch.sigmoid(x)    

    def tick(self):
        raise NotImplementedError("You should implement a tick function in a TrainableStigmergicLayer")
    
    def reset(self):
        raise NotImplementedError("You should implement a reset function in a TrainableStigmergicLayer")
    
    def to(self, *args, **kwargs):
        self = StigmergicModule.to(self, *args, **kwargs) 
        self.init_weights = self.init_weights.to(*args, **kwargs)
        self.bias = self.bias.to(*args, **kwargs)
        self.init_th = self.init_th.to(*args, **kwargs)
        
        return self