import mxnet as mx

def get_model():
    #X = mx.sym.Variable('img1')
    #Y = mx.sym.Variable('img2')
    sim = mx.sym.Variable('sim') 
    #inp = mx.sym.concat(X,Y)
    #fc1 = mx.sym.FullyConnected(data=inp, num_hidden=32)
    #act1 = mx.sym.Activation(data=fc1,act_type='relu')
    #drop1 = mx.sym.Dropout(data= act1,p=0.5)
    #fc2 = mx.sym.FullyConnected(data=drop1, num_hidden=16)
    #act2 = mx.sym.Activation(data=fc2,act_type='relu')
    fc3 = mx.sym.FullyConnected(data=mx.sym.concat(sim), num_hidden=2)
    # softmax loss
    model = mx.sym.SoftmaxOutput(data=fc3, name='softmax')
    return model

model = get_model()

