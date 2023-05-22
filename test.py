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
		return f"Tensor({self.data})"

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

	def sum(self, axis = None):
		out = Tensor(self.data.sum(axis=axis), (self,), 'sum')
		def _backward():
			self.grad += unbroadcast(np.ones_like(self.data) * out.grad, self.data.shape)
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
            graph.node(name = uid , label = f'{n.label} : {n.data} | ∇ : {n.grad}', shape = 'record')
            if n._op:
                graph.node(name = uid + n._op, label = n._op)
                graph.attr('edge',color = 'red',arrowhead="vee")
                graph.edge(str(id(n)) + n._op, str(id(n)))
        for n1, n2 in self.edges:
            if n2._op:
                graph.edge(str(id(n1)), str(id(n2)) + n2._op)
            else:
                graph.edge(str(id(n1)), str(id(n2)))
        return graph
    
    def visualize(self, root, rankdir="LR"):
        self._build_trace(root)
        graph = self._build_graph(rankdir=rankdir)
        return graph

if __name__ == '__main__':
	# from platform import python_version
	# print(python_version())
	# a = Tensor([1,1,1,1], label='a')
	# b = Tensor([2], label='b')
	# c = (a * b).sum()
	# c = (a * b)
	# c.label = 'c'
	# print(c)
	# d = c.sum()
	# d.label = 'd'
	# d.grad = 1.0
	# d._backward()
	# c._backward()
	# a._backward()
	# b._backward()
 
	a = Tensor([[1,2],[3,4],[5,6]], label='a')
	b = Tensor([-1,1], label='b')
	c = (a + b).sum()
	c.label = 'c'
	print(c)
	# d = c.sum()
	# print(f'd {d}')
	# d.label = 'd'
	# c.grad = 1.0
	# d._backward()
	c.backward()
	# a._backward()
	# b._backward()
	print(f'a: {a.data} b: {b.data} c: {c.data}\na.grad:\n{a.grad}\nb.grad:\n{b.grad}\nc.grad:\n{c.grad}')
	visualizer = ForwardGraphVisualizer()
	rankdir = "LR"
	dot = visualizer.visualize( c,rankdir=rankdir)
	dot.render(filename=dot.name,view=True) 