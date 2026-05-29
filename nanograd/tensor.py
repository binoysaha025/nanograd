import numpy as np 

class Tensor:
    def __init__(self, data, _children=(), requires_grad=True):
        self.data = np.array(data, dtype=np.float32)
        self.grad = np.zeros_like(self.data)
        self.requires_grad = requires_grad
        self._backward = lambda: None   # default: do nothing
        self._prev = set(_children)     # what tensors created this one, for computational graph
    
    def matmul(self, other):
        out = Tensor(self.data @ other.data, _children = (self, other))

        def _backward():
            self.grad += out.grad @ other.data.T
            other.grad += self.data.T @ out.grad
        
        out._backward = _backward
        return out
    
    def add(self, other):
        out = Tensor(self.data + other.data, _children = (self, other))

        def _backward():
            self.grad += out.grad
            other.grad += out.grad.sum(axis=0, keepdims=True)
        
        out._backward = _backward
        return out
    
    def relu(self):
        out = Tensor(np.maximum(0, self.data), _children=(self,))

        def _backward():
            self.grad += out.grad * (self.data > 0)
        
        out._backward = _backward
        return out
    
    def log(self):
        out = Tensor(np.log(self.data), _children=(self,))

        def _backward():
            self.grad += out.grad * (1/ self.data)

        out._backward = _backward
        return out
    
    def exp(self):
        out = Tensor(np.exp(self.data), _children=(self,))

        def _backward():
            self.grad += out.grad * np.exp(self.data)

        out._backward = _backward
        return out
    
    def sum(self, axis=None):
        out = Tensor(self.data.sum(axis=axis), _children=(self,))

        def _backward():
            self.grad += np.ones_like(self.data) * out.grad

        out._backward = _backward
        return out

    def backward(self):
        topo = []
        visited = set()

        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        
        build_topo(self)
        self.grad = np.ones_like(self.data)
        
        for node in reversed(topo):
            node._backward()
    