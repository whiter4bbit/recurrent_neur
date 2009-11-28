#linear activation function
linear = lambda x : x
linear_deriv = lambda x: 1

#tangensal sigmoid activation function
sigmoid = lambda x : math.tanh(x)
sigmoid_deriv = lambda x: 1. - x**2
