#!/usr/bin/python3
import math
import random
import numpy as np
from graphviz import Digraph


def unbroadcast(grad:np.ndarray, shape:tuple, to_keep:int=0) -> np.ndarray:
    while len(grad.shape) != len(shape):
        grad = grad.sum(axis=0)
    for i in range(len(shape) - to_keep):
        if grad.shape[i] != shape[i]:
            grad = grad.sum(axis=i, keepdims=True)
    return grad

def to_one_hot(arr, num_classes):
	"""(Freebie) Converts a tensor of classes to one-hot, useful in XELoss

	Example:
	>>> to_one_hot(tensor.Tensor(np.array([1, 2, 0, 0])), 3)
	[[0, 1, 0],
	[0, 0, 1],
	[1, 0, 0],
	[1, 0, 0]]

	Args:
		arr (Tensor): Condensed tensor of label indices
		num_classes (int): Number of possible classes in dataset
						For instance, MNIST would have `num_classes==10`
	Returns:
		Tensor: one-hot tensor
	"""
	arr = arr.data.astype(int)
	a = np.zeros((arr.shape[0], num_classes))
	a[np.arange(len(a)), arr] = 1
	return Tensor(a, True)

def cross_entropy(predicted, target, onehot = False):
	"""Calculates Cross Entropy Loss (XELoss) between logits and true labels.
	For MNIST, don't call this function directly; use nn.loss.CrossEntropyLoss instead.

	Args:
		predicted (Tensor): (batch_size, num_classes) logits
		target (Tensor): (batch_size,) true labels

	Returns:
		Tensor: the loss as a float, in a tensor of shape ()
	"""
	batch_size, num_classes = predicted.shape

	# Tip: You can implement XELoss all here, without creating a new subclass of Function.
	#      However, if you'd prefer to implement a Function subclass you're free to.
	#      Just be sure that nn.loss.CrossEntropyLoss calls it properly.

	# Tip 2: Remember to divide the loss by batch_size; this is equivalent
	#        to reduction='mean' in PyTorch's nn.CrossEntropyLoss
	# see https://stackoverflow.com/questions/44081007/logsoftmax-stability
	x = predicted
	if onehot:
		y = to_one_hot(target,num_classes) 
	else: y = target

	max = Tensor(np.max(x.data,axis=1)) #batchsize x 1
	C = (x.T() - max)
	LSM = (C - C.T().exp().sum(axis=1).log()).T()
	# print(f'LSM shape: {LSM.shape} {(LSM*y).sum()} ')
	# print(f'Tensor(batch_size) {Tensor(batch_size)}')
	Loss = -(LSM*y).sum() / batch_size
	# Loss = -(LSM*y).sum() 

	return Loss
class Tensor:
	def __init__(self, data, _children=(), _op='', label=''):
		if isinstance(data, np.ndarray): self.data = data
		else: self.data = np.array(data)
		self.children = _children
		self.grad = 0.0
		self._backward = lambda: None
		self._prev = set(_children)
		self._op = _op
		self.label = label
  
	def __repr__(self):
		return f"Tensor({self.data}) grad = {self.grad}"

	def __add__(self, other):
		other = other if isinstance(other, Tensor) else Tensor(other)
		out = Tensor(self.data + other.data, (self, other), '+')

		def _backward():
			self.grad += unbroadcast(np.ones_like(self.data) * out.grad, self.data.shape) 
			other.grad += unbroadcast(np.ones_like(other.data) * out.grad, other.data.shape)
		out._backward = _backward
		return out

	def __mul__(self, other):
		other = other if isinstance(other, Tensor) else Tensor(other)
		out = Tensor(self.data * other.data, (self, other), '*')

		def _backward():
			self.grad += unbroadcast(other.data * out.grad, self.data.shape)
			other.grad += unbroadcast(self.data * out.grad, other.data.shape)
		out._backward = _backward
		return out


	def sum(self, axis=None, keepdims=False):
		out = Tensor(self.data.sum(axis=axis, keepdims=keepdims), (self,), 'sum')
		
		def _backward():
			if axis is not None and not keepdims:
				grad_out = np.expand_dims(out.grad, axis=axis)
			else:
				grad_out = out.grad.copy()
			grad = np.ones_like(self.data) * grad_out
			self.grad += grad

		out._backward = _backward
		return out

	def __getitem__(self, idx):
		out = Tensor(self.data[idx], (self,), 'getitem')

		def _backward():
			self.grad += unbroadcast(out.grad, self.data.shape)[idx]
		out._backward = _backward
		return out

	def __pow__(self, other):
		print(f'pow : self {self} other {other}')
		other = other if isinstance(other, Tensor) else Tensor(other)
		out = Tensor(self.data ** other.data, (self, other), '**')

		def _backward():
			self.grad += unbroadcast(other.data * self.data ** (other.data - 1) * out.grad, self.data.shape)
		out._backward = _backward
		return out

	def __neg__(self):
		return self * -1

	def __sub__(self, other):
		return self + (-other)

	def __truediv__(self, other):
		return self * other ** -1
	def __rtruediv__(self, other):
		return other * self ** -1
	def __rmul__(self, other):
		return self * other
	def __radd__(self, other):
		return self + other
	def __rsub__(self, other):
		return other + (-self)

	def tanh(self):
		out = Tensor(np.tanh(self.data), (self,), 'tanh')
		def _backward():
			self.grad += (1 - np.tanh(self.data) ** 2) * out.grad
		out._backward = _backward
		return out

	def sigmoid(self):
		out = Tensor(1 / (1 + np.exp(-self.data)), (self,), 'sigmoid')
		def _backward():
			self.grad += out.data * (1 - out.data) * out.grad
		out._backward = _backward
		return out

	def relu(self):
		out = Tensor(np.maximum(0, self.data), (self,), 'relu')
		def _backward():
			self.grad += (self.data > 0) * out.grad
		out._backward = _backward
		return out

	def exp(self):
		out = Tensor(np.exp(self.data), (self,), 'exp')
		def _backward():
			self.grad += np.exp(self.data) * out.grad
		out._backward = _backward
		return out

	def __matmul__(self, other):
		other = other if isinstance(other, Tensor) else Tensor(other)
		out = Tensor(self.data @ other.data, (self, other), '@')

		def _backward():
			self.grad += out.grad @ other.data.T
			other.grad += self.data.T @ out.grad
		out._backward = _backward
		return out

	def T(self):
		if not len(self.data.shape) == 2: raise Exception("Arg for Transpose must be 2D tensor: {}".format(self.data.shape))
		out = Tensor(self.data.T, (self,), 'T')
		def _backward():
			self.grad += out.grad.T
		out._backward = _backward
		return out

	def reshape(self, shape):
		out = Tensor(self.data.reshape(shape), (self,), 'reshape')
		def _backward():
			self.grad += out.grad.reshape(out.shape)
		out._backward = _backward
		return out

	def log(self):
		out = Tensor(np.log(self.data), (self,), 'log')
		def _backward():
			self.grad += out.grad / self.data
		out._backward = _backward
		return out

	def softmax(self):
		out = Tensor(np.exp(self.data) / np.exp(self.data).sum(axis=-1, keepdims=True), (self,), 'soft_max')
		# def _backward():
		# 	self.grad += out.grad * out.data * (1 - out.data)
		softmax = out.data
		def _backward():
			self.grad += (out.grad - np.reshape(np.sum(out.grad * softmax, 1),[-1, 1])) * softmax
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
		
		self.grad = 1.0
		for node in reversed(topo):
			node._backward()

	@property
	def shape(self):
		"""Returns the shape of the data array in a tuple.
		>>> a = Tensor(np.array([3,2])).shape
		(2,)
		"""
		return self.data.shape

	def visualize(root, rankdir="LR"):
		return ForwardGraphVisualizer().visualize(root,rankdir="LR")


 
class ForwardGraphVisualizer:
    def __init__(self):
        self.nodes, self.edges = set(), set()
    
    def _build_trace(self, node):
        """Performs a recursive depth-first search over the computational graph."""
        if node not in self.nodes:
            if node:
                self.nodes.add(node)
            for child in node.children:
                if type(child).__name__ == "Tensor":
                    self.edges.add((child, node))
                    self._build_trace(child)    
    
    def _build_graph(self, rankdir:str='LR'):
        graph = Digraph(format='png', graph_attr={'rankdir': rankdir})
        for n in self.nodes:
            uid = str(id(n))
            dshape = n.shape if (isinstance(n, Tensor) or isinstance(n, np.ndarray)) and  (len(n.shape) != 0) else n
            gshape = n.grad.shape if isinstance(n.grad, Tensor) or isinstance(n.grad, np.ndarray) else n.grad
            graph.node(name = uid , label =f'<<table border="1" cellborder="0" cellspacing="1"><tr><td BGCOLOR= "deepskyblue">{n.label}: {dshape}</td>\
                							 </tr><tr><td BGCOLOR= "brown1">{gshape}</td></tr></table>>', shape = 'plaintext')
            if n._op:
                graph.node(name = uid + n._op, label = n._op)
                graph.edge(str(id(n)) + n._op, str(id(n)), color = 'red',arrowhead="vee",dir="back")
                graph.edge(str(id(n)) + n._op, str(id(n)),color = 'blue',arrowhead="vee")
        for n1, n2 in self.edges:
            if n2._op:
                graph.edge(str(id(n1)), str(id(n2)) + n2._op, color = 'red',arrowhead="vee",dir="back")
                graph.edge(str(id(n1)), str(id(n2)) + n2._op,color = 'blue',arrowhead="vee")

            else:
                graph.edge(str(id(n1)), str(id(n2)))
        return graph
    
    def visualize(self, root, rankdir="LR"):
        self._build_trace(root)
        graph = self._build_graph(rankdir=rankdir)
        return graph
    

