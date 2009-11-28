import math

class Node(object):
    def __init__(self, layer):
        self.layer = layer
        self.inputs = []
        self._value = 0.0
        self.error = 0.0

    def set_value(self, value):
        self._value = value

    def get_value(self):
        return self._value

    def forward(self):
        self._value = 0.0
        for con in self.inputs:
            self._value += con.w*con.backw.activate()

    def activate(self):
        return self.layer.func(self.get_value())

    def propagate(self):
        self.error *= self.layer.derivative(self.activate())
        for con in self.inputs:
            con.backw.error += con.w*self.error

    def can_be_forward(self):
        return True

    def update_weights(self, nu):
        for con in self.inputs:
            err = self.error*con.backw.activate()
            con.w+= nu*err

from random import random

class Connection(object):
    def __init__(self, backw, forw):
        self.backw = backw
        self.forw = forw

    def randomize(self):
        self.w = .4*random() - .2
