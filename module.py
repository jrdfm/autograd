#! /usr/bin/env python3
from tensor import *
import random


class Module:
    """Base class (superclass) for all components of an NN.
    https://pytorch.org/docs/stable/generated/torch.nn.Module.html

    Layer classes and even full Model classes should inherit from this Module.
    Inheritance gives the subclass all the functions/variables below

    NOTE: You shouldn't ever need to instantiate Module() directly."""

    def __init__(self):
        self._submodules = {} # Submodules of the class
        self._parameters = {} # Trainable params in module and its submodules

        self.is_train = True # Indicator for whether or not model is being trained.

    def train(self):
        """Activates training mode for network component"""
        self.is_train = True

    def eval(self):
        """Activates evaluation mode for network component"""
        self.is_train = False

    def forward(self, *args):
        """Forward pass of the module"""
        raise NotImplementedError("Subclasses of Module must implement forward")

    def is_parameter(self, obj):
        """Checks if input object is a Tensor of trainable params"""
        return isinstance(obj, Tensor) 
    
    def parameters(self):
        """Returns an interator over stored params.
        Includes submodules' params too"""
        for name, parameter in self._parameters.items():
            yield parameter
        for name, module in self._submodules.items():
            for parameter in module.parameters():
                yield parameter

    def add_parameter(self, name, value):
        """Stores params"""
        self._parameters[name] = value

    def add_module(self, name, value):
        """Stores module and its params"""
        self._submodules[name] = value

    def add_tensor(self, name, value) -> None:
        """Stores tensors"""
        self._tensors[name] = value

    def __setattr__(self, name, value):
        """Magic method that stores params or modules that you provide"""
        if self.is_parameter(value):
            self.add_parameter(name, value)
        elif isinstance(value, Module):
            self.add_module(name, value)
        elif isinstance(value, Tensor):
            self.add_tensor(name, value)


        object.__setattr__(self, name, value)

    def __call__(self, *args):
        """Runs self.forward(args). Google 'python callable classes'"""
        return self.forward(*args)


class Linear(Module):
	"""A linear layer (aka 'fully-connected' or 'dense' layer)

	>>> layer = Linear(2,3)
	>>> layer(Tensor.ones(10,2)) # (batch_size, in_features)
	<some tensor output with size (batch_size, out_features)>

	Args:
		in_features (int): # dims in input
							(i.e. # of inputs to each neuron)
		out_features (int): # dims of output
							(i.e. # of neurons)

	Inherits from:
		Module (mytorch.nn.module.Module)
	"""
	def __init__(self, in_features, out_features):
		super().__init__()

		self.in_features = in_features
		self.out_features = out_features
		# Randomly initializing layer weights
		k = 1 / in_features
		self.weight = Tensor(k * (np.random.rand(out_features, in_features) - 0.5),label="linear_weight")
		self.bias = Tensor(k * (np.random.rand(out_features) - 0.5),label = "linear_bias")
		
  
	def __repr__(self):
		return f"Linear({self.in_features}, {self.out_features})"

	# def parameters_(self):
	# 	return [self.weight, self.bias]

	def forward(self, x):
		"""
		Args:
			x (Tensor): (batch_size, in_features)
		Returns:
			Tensor: (batch_size, out_features)
		"""
		if not (type(x).__name__ == 'Tensor' or type(self.weight).__name__ == 'Tensor'):
			raise Exception(f"X must be Tensor. Got {type(x)}")

		output = x @ self.weight.T() + self.bias
		return output


class Sequential(Module):
	"""Passes input data through stored layers, in order

	>>> model = Sequential(Linear(2,3), ReLU())
	>>> model(x)
	<output after linear then relu>

	Inherits from:
		Module (nn.module.Module)
	"""

	def __init__(self, *layers):
		super().__init__()
  
		self.layers = layers
		self.children = self._parameters
		# iterate through args provided and store them
		for idx, l in enumerate(self.layers):
			self.add_module(str(idx), l)
      
	def __repr__(self):
		return f"Sequential({', '.join([str(l) for l in self.layers])})"

	def __iter__(self):
		"""Enables list-like iteration through layers"""
		yield from self.layers

	def __getitem__(self, idx):
		"""Enables list-like indexing for layers"""
		return self.layers[idx]

	# def parameters_(self):
	# 	return [p for layer in self.layers for p in layer.parameters()]

	def forward(self, x):
		"""Passes input data through each layer in order
		Args:
			x (Tensor): Input data
		Returns:
			Tensor: Output after passing through layers
		"""
		for layer in self.layers:
			x = layer(x)
		return x
    
class ReLU(Module):
	def __init__(self):
			super().__init__()
	def forward(self, x):
		return x.relu()

class Tanh(Module):
	def __init__(self):
			super().__init__()
	def forward(self, x):
		return x.tanh()

class Sigmoid(Module):
	def __init__(self):
			super().__init__()
	def forward(self, x):
		return x.sigmoid()
    
class CrossEntropyLoss(Module):
    """The XELoss function.
    This class is for human use; just calls function in nn.functional.
    Does not need args to initialize.

    >>> criterion = CrossEntropyLoss()
    >>> criterion(outputs, labels)
    3.241
    """
    def __init__(self):
        pass

    def forward(self, predicted, target):
        """
        Args:
            predicted (Tensor): (batch_size, num_classes)
            target (Tensor): (batch_size,)

        Returns:
            Tensor: loss, stored as a float in a tensor
        """
        # Simply calls nn.functional.cross_entropy
        # If you implement your own Function subclass you may need to modify this"""
        return cross_entropy(predicted, target)
    
class Optimizer():
    """Base class for optimizers. Shouldn't need to modify."""
    def __init__(self, params):
        self.params = list(params)
        self.state = [] # Technically supposed to be a dict in real torch

    def step(self):
        """Called after generating gradients; updates network weights."""
        raise NotImplementedError

    def zero_grad(self):
        """After stepping, you need to call this to reset gradients.
        Otherwise they keep accumulating."""
        for param in self.params:
            param.grad = 0.0
            
class SGD(Optimizer):
	"""Stochastic Gradient Descent optimizer.

	>>> optimizer = SGD(model.parameters(), lr=0.1, momentum=0.9)

	Args:
		params (list or iterator): <some module>.parameters()
		lr (float): learning rate (eta)
		momentum (float): momentum factor (beta)

	Inherits from:
		Optimizer (optim.optimizer.Optimizer)
	"""
	def __init__(self, params, lr=0.001, momentum=0.0):
		super().__init__(params) # inits parent with network params
		self.lr = lr
		self.momentum = momentum

		# This tracks the momentum of each weight in each param tensor
		self.momentums = [np.zeros(t.shape) for t in self.params]

	def step(self):
		"""Updates params based on gradients stored in param tensors"""
		for i, p in enumerate(self.params):
			m = self.momentums
			m[i] = (self.momentum * m[i]) - (self.lr * p.grad)
			p.data = p.data + m[i]
