import torch
from ..stigmergic_module import StigmergicModule

class TemporalAdapter(StigmergicModule):
    def __init__(self, inputs, time_ticks, **kwargs):
        StigmergicModule.__init__(self, inputs, inputs*time_ticks, **kwargs)
        self.n_inputs = inputs
        self.time_ticks = time_ticks
        self.i = 0
        self.device = torch.device("cpu")
        self.memory = None

    def init_memory(self, batch_size):
        self.memory = torch.zeros(batch_size, self.time_ticks*self.n_inputs, device=self.device)

    def forward(self, input):
        if self.memory is None:
            self.init_memory(input.shape[0])
        self.memory[:,self.i*self.n_inputs:(self.i+1)*self.n_inputs] = input
        self.i = (self.i + 1) % self.time_ticks
        return self.memory

    def tick(self):
        pass

    def reset(self):
        self.memory = None
        self.i = 0

    def to(self, *args, **kwargs):
        self = StigmergicModule.to(self, *args, **kwargs)
        if self.memory is not None:
            self.memory = self.memory.to(*args, **kwargs)
        self.device = torch.tensor(0).to(*args, **kwargs).device

        return self