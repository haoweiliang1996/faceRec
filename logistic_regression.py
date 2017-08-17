import mxnet as mx

def get_model():
    X = mx.sym.Variable('img1')
    Y = mx.sym.Variable('img2')
    inp = mx.sym.concat(X,Y)
    fc1 = mx.sym.FullyConnected(data=inp, num_hidden=500)
    sim = mx.symbol.dot(X,Y) / (mx.sym.dot(X,X)**0.5*mx.sym.dot(Y,Y)**0.5)
    fc2 = mx.sym.FullyConnected(data=mx.sym.concat(fc1,sim), num_hidden=2)
    # softmax loss
    model = mx.sym.SoftmaxOutput(data=fc2, name='softmax')
    return model

model = get_model()

