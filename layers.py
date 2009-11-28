from nodes import Node, Connection
from utils import linear, linear_deriv

class Context(object):
    def __init__(self, inputs, targets):
        self._inputs = inputs
        self._targets = targets

    def get_inputs(self):
        return self._inputs

    def get_targets(self):
        return self._targets

class Layer(object):
    def __init__(self, nodes_num,  name='',
                 func = linear,derivative = linear_deriv):
        self.nodes = [Node(self) for node in xrange(nodes_num) ]
        self.name = name
        self.func = func
        self.derivative = derivative
        self.inputs = []
        self.outputs = []

    def set_values(self, context):
        for i in xrange(len(context.get_inputs())):
            self.nodes[i].set_value(context.get_inputs()[i])

    def set_weights(self, weights):
        for i in xrange(len(self.nodes)):
            for j in range(len(weights)):
                self.nodes[i].inputs[j].w = weights[j][i]

    def set_targets(self, context):
        pass

    def get_values(self):
        return [node.activate() for node in self.nodes]

    def forward(self):
        for node in self.nodes:
            node.forward()

    def get_weights(self):
        return [[con.w for con in node.inputs] for node in self.nodes]

    def connect(self, layer):
        for node in self.nodes:
            for forw_node in layer.nodes:
                con = Connection(node, forw_node)
                if forw_node.can_be_forward():
                    layer.inputs.append(con)
                    self.outputs.append(con)
                    forw_node.inputs.append(con)

    def propagate(self):
        for node in self.nodes:
            node.propagate()

    def update_weights(self, nu):
        for node in self.nodes:
            node.update_weights(nu)

    def randomize(self):
        for con in self.inputs:
            con.randomize()

class ContextLayer(Layer):
    def __init__(self, nodes_num, name = '', func = linear, derivative = linear_deriv):
        Layer.__init__(self, nodes_num, name, func, derivative)
        for i in xrange(nodes_num):
            self.nodes[i].set_value(0.0)

class HiddenLayer(Layer):
    def __init__(self, nodes_num, name = '', func = linear, derivative = linear_deriv, context_layer = None):
        Layer.__init__(self, nodes_num, name, func, derivative)
        self._context_layer = context_layer

    def copy(self):
        if self._context_layer:
            for i in xrange(len(self.nodes)):
                if self.nodes[i].can_be_forward():
                    self._context_layer.nodes[i].set_value( self.nodes[i].activate() )

class OutputLayer(Layer):
    def __init__(self, nodes_num, name = '', func = linear, derivative = linear_deriv):
        Layer.__init__(self, nodes_num, name, func, derivative)

    def set_targets(self, context):
        self.targets = context.get_targets()

    def propagate(self):
        for i in xrange(len(self.nodes)):
            self.nodes[i].error = self.targets[i]-self.nodes[i].activate()
            self.nodes[i].propagate()
