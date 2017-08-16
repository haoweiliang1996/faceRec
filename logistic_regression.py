import mxnet as mx

def get_model():
    X = mx.sym.Variable('data')
    fc2 = mx.sym.FullyConnected(data=X, num_hidden=2)
    # softmax loss
    model = mx.sym.SoftmaxOutput(data=fc2, name='softmax')
    return model

model = get_model()

