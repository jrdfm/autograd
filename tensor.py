#!/usr/bin/python3
import math
import random
import numpy as np
from graphviz import Digraph


def unbroadcast(grad: np.ndarray, shape: tuple, to_keep: int = 0) -> np.ndarray:
    if grad.shape == shape: return grad
    diff_dims = len(shape) - len(grad.shape)
    print(f'diff_dims {diff_dims} shape {shape} grad.shape {grad.shape}')
    if diff_dims > 0: grad = grad.sum(axis=0)
    for i in range(diff_dims, len(shape) - to_keep):
        print(f'i {i} grad.shape[{i}] {grad.shape[i]} shape[{i}] {shape[i]}')
        if grad.shape[i] != shape[i]:
            grad = grad.sum(axis=i, keepdims=True)
            print(f'grad {grad}\n')
    return grad

def unbroadcast(grad:np.ndarray, shape:tuple, to_keep:int=0) -> np.ndarray:
    while len(grad.shape) != len(shape):
        grad = grad.sum(axis=0)
    for i in range(len(shape) - to_keep):
        if grad.shape[i] != shape[i]:
            grad = grad.sum(axis=i, keepdims=True)
    return grad

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
		# print(f'__add__ self {self.shape} other {other.label} {other.shape} type {type(other)}')
		# print(f'__add__ other {other}')
		other = other if isinstance(other, Tensor) else Tensor(other)
		out = Tensor(self.data + other.data, (self, other), '+')

		def _backward():
			# print(f'__add__ _backward self {self.grad} other {other.grad} ')
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

	def reduce_sum(self, axis=None):
		out = Tensor(self.data.sum(axis=axis), (self,), 'reduce_sum')
		def _backward():
			
			output_shape = np.array(self.data.shape)
			output_shape[axis] = 1
			tile_scaling = self.data.shape // output_shape
			grad = np.reshape(out.grad, output_shape)
			self.grad += np.tile(grad, tile_scaling)
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
            shape = n.grad.shape if isinstance(n.grad, Tensor) or isinstance(n.grad, np.ndarray) else n.grad
            graph.node(name = uid , label =f'<<table border="1" cellborder="0" cellspacing="1"><tr><td BGCOLOR= "deepskyblue">{n.label}: {n.data.shape}</td>\
                							 </tr><tr><td BGCOLOR= "brown1">{shape}</td></tr></table>>', shape = 'plaintext')
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
    

