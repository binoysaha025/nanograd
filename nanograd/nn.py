import numpy as np
from nanograd.tensor import Tensor

class Module:
    def parameters(self):
        return []
    def __call__(self, *args):
        return self.forward(*args)
    def forward(self, *args):
        raise NotImplementedError
    
class Linear(Module):
    def __init__(self, in_features, out_features):
        self.W = Tensor(np.random.randn(in_features, out_features) * 0.01)
        self.b = Tensor(np.zeros((1, out_features)))
    
    def forward(self, x):
        return x.matmul(self.W).add(self.b)
    
    def parameters(self):
        return [self.W, self.b]

class ReLU(Module):
    def forward(self, x):
        return x.relu()
    
    def parameters(self):
        return []
    
class Sequential(Module):
    def __init__(self, *layers):
        self.layers = layers
    
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
    
    def parameters(self):
        params = []
        for layer in self.layers:
            params += layer.parameters()
        return params
    
class CrossEntropyLoss(Module):
    def forward(self, logits, y):
        # stable softmax - subtract max to prevent exp overflow
        shifted = logits.data - logits.data.max(axis=1, keepdims=True)
        exp_logits = np.exp(shifted)
        probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
        
        batch_size = y.shape[0]
        
        # get correct class probabilities
        correct_probs = probs[np.arange(batch_size), y]
        
        # compute loss directly
        log_probs = np.log(correct_probs + 1e-8)
        loss_val = -log_probs.mean()
        
        out = Tensor(loss_val)
        
        def _backward():
            grad = probs.copy()
            grad[np.arange(batch_size), y] -= 1
            grad /= batch_size
            logits.grad += grad
        
        out._backward = _backward
        out._prev = {logits}
        return out