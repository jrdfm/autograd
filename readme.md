# autograd

Automatic differentiation engine that is similar to PyTorch, used to automatically calculate the derivative
/ gradient of any computable function.
Inspired by Karphaty's [micrograd](https://github.com/karpathy/micrograd)

# Features

- PyTorch-like backpropagation, reverse autodifferentiation,  on dynamically built computational graph, DAG.
- Activations: ReLU, Sigmoid, tanh
- Loss: CrossEntropyLoss, Mean squared error
- Layers: Linear, Sequential
- Optimizers: SGD
- Computational graph visualizer 


# Todo
- Layers: BatchNorm1d, BatchNorm2d, Flatten, Dropout
- Convolutions: Conv1d, Conv2d, MaxPool2d, AvgPool2d
- Weight initialization: Glorot uniform, Glorot normal, Kaiming uniform, Kaiming normal
- Activations: Swish, ELU, LeakyReLU
- Optimizers: AdamW, Adam
- RNN, GRU