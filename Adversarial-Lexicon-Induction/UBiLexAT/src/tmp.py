import theano
import theano.tensor as T
import lasagne


gen_input_var = T.matrix('gen_input_var')
print "G",gen_input_var
W=lasagne.init.Orthogonal()
print "W",W
