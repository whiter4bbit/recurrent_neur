from netw import NeuralNetw
from layers import Context

class SequencePrediction(object):
    def __init__(self, seq, n, k):
        self.seq = seq
        self.n = n
        self.k = k

    def normalize_one(self, a):
        return a
#        return 2.*a/100.

    def denormalize_one(self, a):
        return a
#        return a*100./2.

    def normalize(self, list_):
        return [self.normalize_one(e) for e in list_]

    def denormalize(self, list_):
        return [self.denormalize_one(e) for e in list_]

    def get_sample(self):
        sample = []
        for i in xrange(len(self.seq)-(self.n+self.k)+1):
            sample.append((self.seq[i:i+self.n], self.seq[i+self.n:i+self.n+self.k]))
        return sample

    def prepare(self, iterations):
        self.nw = NeuralNetw()
        self.nw.init_layers(self.n, self.n+self.k, self.k)
        self.nw.randomize()
        self.nw.train(iterations, self.get_sample())

    def predict(self):
        self.prepare(5000)
        ask_for = self.seq[-(self.n+self.k-1):]
        if self.k>1:
            ask_for = ask_for[:-(self.k-1)]
        print "asking for: %s" % (ask_for)
        last = self.normalize(ask_for)
        return self.denormalize(self.nw.process_sample(Context(last, [])))

import math

if __name__=="__main__":
    fib_seq = [1,1,2,3,5,8,13,21]
    print "Fibonacci: %s" % str(fib_seq)
    p = SequencePrediction(fib_seq, 3, 2)
    print "samples: %s" % str(p.get_sample())
    print p.predict()

    sin_sample = [math.sin(math.pi/180.*a) for a in xrange(0, 360, 15)]
    sin_sample += [math.sin(math.pi/180.*15), math.sin(math.pi/180.*30)]
    print "Sin: %s" % str(sin_sample)
    p = SequencePrediction(sin_sample, 3, 2)
    print "samples: %s" % str(p.get_sample())
    print p.predict()
    
