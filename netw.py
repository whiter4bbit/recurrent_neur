from nodes import *
from utils import *
from layers import *

class NeuralNetw(object):
    def __init__(self):
        self.layers = []

    def init_layers(self, input, hidden, output):
        self.layers = []
        self.input_layer = Layer(input, 'input')
        self.context_layer = ContextLayer(hidden, 'context')
        self.hidden_layer = HiddenLayer(hidden, 'hidden',
                                                             context_layer=self.context_layer)
        self.output_layer = OutputLayer(output, 'output')
        self.layers = [self.input_layer,
                             self.context_layer,
                             self.hidden_layer,
                             self.output_layer]
        self.input_layer.connect(self.hidden_layer)
        self.context_layer.connect(self.hidden_layer)
        self.hidden_layer.connect(self.output_layer)

    def process_sample(self, context):
        self.input_layer.set_values(context)
        self.hidden_layer.forward()
        self.output_layer.forward()
        return self.output_layer.get_values()

    def reset_errors(self):
        for node in self.output_layer.nodes:
            node.error = 0
        for node in self.hidden_layer.nodes:
            node.error = 0

    def learn_step(self, context):
        res = self.process_sample(context)
        self.output_layer.set_targets(context)
        self.reset_errors()
        self.output_layer.propagate()
        self.hidden_layer.propagate()
        a = 0.2
        adapt_a_ = 4./(sum((i*100.)**2 for i in res)+1)
        adapt_a = 4./(sum((i*100.)**2 for i in context.get_inputs())+1)
        self.hidden_layer.update_weights(adapt_a)
        self.output_layer.update_weights(adapt_a_)
        self.hidden_layer.copy()

        return 0.5*sum([math.pow(context.get_targets()[i]-res[i],2) for i in xrange(len(res))])

    def randomize(self):
        for layer in self.layers:
            layer.randomize()

    def normalize_one(self, a):
        return 2.*a/100.

    def denormalize_one(self, a):
        return a*100./2.

    def normalize(self, list_):
        return [self.normalize_one(e) for e in list_]

    def denormalize(self, list_):
        return [self.denormalize_one(e) for e in list_]

    def train(self, epochs, samples, min_error=0.000001):
        mse = 1e100
        epoch = 0
        while mse>min_error and epoch<epochs:
            mse = 0.0
            for sample in samples:
                mse += self.learn_step(Context(self.normalize(sample[0]), self.normalize(sample[1])))
            if epoch%300==0:
                print "%-14f" % mse
            epoch+=1
