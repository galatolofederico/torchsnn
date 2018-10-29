# torchsnn
pytorch implementation of the Stigmergic Neural Networks as presented in the paper [*Using stigmergy to incorporate the time into artificial neural networks*](http://www.iet.unipi.it/m.cimino/publications/cimino_pub62.pdf).

This package was wrote with the intent to make as **easy** as possible to integrate the Stigmergic Neural Networks into the existing models.

You can safely **mix** native pytorch Modules with ours.  
The only **catch** is that you should use *StigmergicModule* (which extends pytorch's *Module*) as base class for your models in order to be able to *tick()* and *reset()* them.

Implementing our [proposed architecture to solve MNIST]() becomes as easy as:
```python
import torch
import torchsnn

net = torchsnn.Sequential(
    torchsnn.SimpleLayer(28, 10),
    torchsnn.FullLayer(10, 10),
    torchsnn.TemporalAdapter(10, 28),
    torch.nn.Linear(10*28, 10),
    torch.nn.Sigmoid()
)
```

You can train a *StigmergicModule* as you would do with a pytorch's *Module*, but don't forget to *reset()* and *tick()* it!

```python
optimizer = torch.optim.Adam(net.parameters(), lr = 0.001)

for i in range(0,N):
    for X,Y in zip(dataset_X, dataset_Y):
        net.reset()
        out = None
        for xi in X:
            out = net(torch.tensor(xi, dtype=torch.float32))
            net.tick()
        
        loss = (Y-out)**2
        
        loss.backward()
        optimizer.step()
```

### Does it support batch inputs?

Yes! if you forward into a StigmergicModule a batch of inputs it will return a batch of outputs

```python
for t in range(0, num_ticks):
    batch_out[0], batch_out[1], ... = net(torch.tensor([batch_in[0][t], batch_in[1][t], ...]))
    net.tick()

```
### Can it run on CUDA?

Yes and as you will expect from a pytorch Module!  
You just need to call the *to(device)* method on a model to move it in the GPU memory

```python
device = torch.device("cuda")

net = net.to(device)

net.forward(torch.tensor(..., device=device))
```

### How do I install it?

torchsnn is on PyPI so you just need to run

```
pip install torchsnn
```

## Documentation

### torchsnn.StigmergicModule

Base class for a stigmergic network or layer.  
If you are writing your own *StigmergicModule* you have to implement the functions
* tick()
* reset()

If you are using others *StigmergicModule* it will propagate these calls in its subtree.  
For example if you want to build a network with a *Linear* and a *SimpleLayer* you can do something like:

```python
import torch
import torchsnn

class TestNetwork(torchsnn.StigmergicModule):
    def __init__(self):
        torchsnn.StigmergicModule.__init__(self)
        self.linear = torch.nn.Linear(2,5)
        self.stigmergic = torchsnn.SimpleLayer(5,2)

    def forward(self, input):
        l1 = torch.sigmoid(self.linear(input))
        l2 = self.stigmergic(l1)
        return l2

net = TestNetwork()
```

### torchsnn.Sequential

Function with the same interface of *torch.nn.Sequential* for building sequential networks.  
The same network of the previous example can be built with:
```python
import torch
import torchsnn

net = torchsnn.Sequential(
    torch.nn.Linear(2,5),
    torch.nn.Sigmoid(),
    torchsnn.SimpleLayer(5,2)
)
```

### torchsnn.SimpleLayer

It this layer only the thresholds are stigmergic variables and their *stimuli* are the output values.  

![](https://latex.codecogs.com/gif.latex?x%28t%29%20%3D%20%5Ctext%7Binput%20at%20time%20t%7D)  

![](https://latex.codecogs.com/gif.latex?y%28t%29%20%3D%20%5Ctext%7Boutput%20at%20time%20t%7D)  

![](https://latex.codecogs.com/gif.latex?th%28t%29%20%3D%20%5Ctext%7Bthreshold%20at%20time%20t%7D)

<br>

![](https://latex.codecogs.com/gif.latex?y%28t%29%20%3D%20%5Csigma%28Wx%28t%29%20&plus;%20b%20-%20th%28t%29%29)  

![](https://latex.codecogs.com/gif.latex?th%28t%29%20%3D%20C%28th%28t-1%29%20&plus;%20My%28t-1%29%20-%20%5Ctau%2C%20min%2C%20max%29) 

### torchsnn.FullLayer

In this layer both thresholds and weights are stigmergic variables and their *stimuli* are respectively the output values and the input ones.  

![](https://latex.codecogs.com/gif.latex?x%28t%29%20%3D%20%5Ctext%7Binput%20at%20time%20t%7D)  

![](https://latex.codecogs.com/gif.latex?y%28t%29%20%3D%20%5Ctext%7Boutput%20at%20time%20t%7D)  

![](https://latex.codecogs.com/gif.latex?th%28t%29%20%3D%20%5Ctext%7Bthreshold%20at%20time%20t%7D) 

![](https://latex.codecogs.com/gif.latex?W%28t%29%20%3D%20%5Ctext%7Bweights%20at%20time%20t%7D)  

<br>

![](https://latex.codecogs.com/gif.latex?y%28t%29%20%3D%20%5Csigma%28Wx%28t%29%20&plus;%20b%20-%20th%28t%29%29)  

![](https://latex.codecogs.com/gif.latex?th%28t%29%20%3D%20C%28th%28t-1%29%20&plus;%20My%28t-1%29%20-%20%5Ctau%2C%20min%2C%20max%29)  

![](https://latex.codecogs.com/gif.latex?J%5Em%28v%29%20%3D%20%5Cbegin%7Bbmatrix%7D%20v_%7B0%7D%20%26%20v_%7B0%7D%20%26%20%5Cdots%20%26%20v_%7B0%7D%20%5C%5C%20v_%7B1%7D%20%26%20v_%7B1%7D%20%26%20%5Cdots%20%26%20v_%7B1%7D%20%5C%5C%20%5Cvdots%20%26%20%5Cvdots%20%26%20%5Cddots%20%26%20%5Cvdots%20%5C%5C%20v_%7Bn-1%7D%20%26%20v_%7Bn-1%7D%20%26%20%5Cdots%20%26%20v_%7Bn-1%7D%20%5Cend%7Bbmatrix%7D)  

![](https://latex.codecogs.com/gif.latex?W%28t%29%20%3D%20C%28W%28t-1%29%20&plus;%20J%5Em%28X%28t-1%29%29M_w%20-%20%5Ctau_w%2C%20min%2C%20max%29) 

## Citing

We can't wait to see what you will build with the Stigmergic Neural Networks!  
When you will publish your work you can use this BibTex to cite us :)

```
@article{galatolo_snn
,	author	= {Galatolo, Federico A and Cimino, Mario GCA and Vaglini, Gigliola}
,	title	= {Using stigmergy to incorporate the time into artificial neural networks}
,	journal	= {MIKE 2018}
,	year	= {2018}
}
```

## Contributing

This code is released under GNU/GPLv3 so feel free to fork it and submit your changes, every PR helps.  
If you need help using it or for any question please reach me at [galatolo.federico@gmail.com](mailto:galatolo.federico@gmail.com) or on Telegram  [@galatolo](https://t.me/galatolo)